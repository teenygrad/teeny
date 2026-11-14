/*
 * Copyright (c) 2026 Teenygrad.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

//! SSA construction for the MLIR blocks of a function: where each MIR local needs a block
//! argument (a phi), and an order to lower the blocks in.
//!
//! MIR locals are mutable variables; MLIR values are assigned once. A local assigned on more than
//! one path into a block, and still live there, needs a block argument at that block, whose
//! operand on each incoming branch is the local's value at the end of that predecessor. Those
//! blocks are the iterated dominance frontier of the blocks that assign the local (Cytron et al.),
//! pruned to where the local is live. Lowering the blocks in reverse postorder, each starting
//! from the values at the end of its immediate dominator, then gives every use a dominating
//! definition, across branches and loop back edges alike (teenyc-3a1).

// The hash maps here are only iterated where order can't change the result: fixpoints over sets,
// and phi lists that are sorted before use.
#![allow(rustc::potential_query_instability)]

use std::collections::{HashMap, HashSet};

use rustc_middle::mir::visit::{MutatingUseContext, PlaceContext, Visitor};
use rustc_middle::mir::{BasicBlock, Body, Local, Location};

/// A control-flow graph over dense node indices, entered at node 0.
pub(crate) struct Graph {
    succs: Vec<Vec<usize>>,
}

impl Graph {
    pub(crate) fn new(succs: Vec<Vec<usize>>) -> Self {
        Graph { succs }
    }

    fn len(&self) -> usize {
        self.succs.len()
    }

    fn preds(&self) -> Vec<Vec<usize>> {
        let mut preds = vec![Vec::new(); self.len()];
        for (node, succs) in self.succs.iter().enumerate() {
            for &succ in succs {
                preds[succ].push(node);
            }
        }
        preds
    }

    /// The nodes reachable from the entry, in reverse postorder: every node comes after its
    /// immediate dominator, and after all its predecessors except along back edges.
    pub(crate) fn reverse_postorder(&self) -> Vec<usize> {
        let mut visited = vec![false; self.len()];
        let mut postorder = Vec::with_capacity(self.len());
        // An explicit stack of (node, next successor to visit), so deep CFGs can't overflow.
        let mut stack = vec![(0, 0)];
        visited[0] = true;
        while let Some((node, next)) = stack.last_mut() {
            if let Some(&succ) = self.succs[*node].get(*next) {
                *next += 1;
                if !visited[succ] {
                    visited[succ] = true;
                    stack.push((succ, 0));
                }
            } else {
                postorder.push(*node);
                stack.pop();
            }
        }
        postorder.reverse();
        postorder
    }

    /// Each reachable node's immediate dominator (the entry's is itself), by the iterative
    /// algorithm of Cooper, Harvey and Kennedy. `None` for unreachable nodes.
    pub(crate) fn immediate_dominators(&self, rpo: &[usize]) -> Vec<Option<usize>> {
        let mut rpo_index = vec![usize::MAX; self.len()];
        for (i, &node) in rpo.iter().enumerate() {
            rpo_index[node] = i;
        }
        let preds = self.preds();
        let mut idom: Vec<Option<usize>> = vec![None; self.len()];
        idom[0] = Some(0);
        let intersect = |idom: &[Option<usize>], mut a: usize, mut b: usize| {
            while a != b {
                while rpo_index[a] > rpo_index[b] {
                    a = idom[a].expect("processed node has a dominator");
                }
                while rpo_index[b] > rpo_index[a] {
                    b = idom[b].expect("processed node has a dominator");
                }
            }
            a
        };
        let mut changed = true;
        while changed {
            changed = false;
            for &node in &rpo[1..] {
                let mut new_idom = None;
                for &pred in &preds[node] {
                    if idom[pred].is_none() {
                        continue;
                    }
                    new_idom = Some(match new_idom {
                        None => pred,
                        Some(current) => intersect(&idom, pred, current),
                    });
                }
                if new_idom.is_some() && idom[node] != new_idom {
                    idom[node] = new_idom;
                    changed = true;
                }
            }
        }
        idom
    }

    /// Each node's dominance frontier: the nodes where its dominance ends, i.e. that it doesn't
    /// strictly dominate but one of whose predecessors it dominates.
    pub(crate) fn dominance_frontiers(&self, idom: &[Option<usize>]) -> Vec<HashSet<usize>> {
        let mut frontiers = vec![HashSet::new(); self.len()];
        for (node, preds) in self.preds().iter().enumerate() {
            let Some(node_idom) = idom[node] else { continue };
            if preds.len() < 2 {
                continue;
            }
            for &pred in preds {
                if idom[pred].is_none() {
                    continue;
                }
                let mut runner = pred;
                while runner != node_idom {
                    frontiers[runner].insert(node);
                    runner = idom[runner].expect("reachable node has a dominator");
                }
            }
        }
        frontiers
    }
}

/// The iterated dominance frontier of `defs`: the nodes where values from two or more of them
/// can meet, so a variable assigned in `defs` may need a phi there.
pub(crate) fn iterated_frontier(
    frontiers: &[HashSet<usize>],
    defs: impl IntoIterator<Item = usize>,
) -> HashSet<usize> {
    let mut phis = HashSet::new();
    let mut worklist: Vec<usize> = defs.into_iter().collect();
    let mut queued: HashSet<usize> = worklist.iter().copied().collect();
    while let Some(node) = worklist.pop() {
        for &frontier in &frontiers[node] {
            if phis.insert(frontier) && queued.insert(frontier) {
                worklist.push(frontier);
            }
        }
    }
    phis
}

/// The locals a block reads before writing them (`uses`) and the locals it assigns whole
/// (`defs`). An assignment through a projection (`_1.0 = ..`) reads the rest of the local, so it
/// is a use, not a def.
#[derive(Default)]
pub(crate) struct UseDef {
    pub(crate) uses: HashSet<Local>,
    pub(crate) defs: HashSet<Local>,
}

/// The locals one statement or terminator reads and assigns whole. Collected per statement,
/// because the visitor walks an assignment's destination before its operands.
#[derive(Default)]
struct LocalAccesses {
    reads: HashSet<Local>,
    writes: HashSet<Local>,
}

impl<'tcx> Visitor<'tcx> for LocalAccesses {
    fn visit_local(&mut self, local: Local, context: PlaceContext, _location: Location) {
        match context {
            PlaceContext::MutatingUse(
                MutatingUseContext::Store
                | MutatingUseContext::Call
                | MutatingUseContext::AsmOutput
                | MutatingUseContext::Yield,
            ) => {
                self.writes.insert(local);
            }
            context if context.is_use() => {
                self.reads.insert(local);
            }
            _ => {}
        }
    }
}

/// The uses and defs of block `bb`.
pub(crate) fn block_use_def<'tcx>(mir: &Body<'tcx>, bb: BasicBlock) -> UseDef {
    let data = &mir.basic_blocks[bb];
    let mut use_def = UseDef::default();
    let mut record = |accesses: LocalAccesses| {
        for local in accesses.reads {
            if !use_def.defs.contains(&local) {
                use_def.uses.insert(local);
            }
        }
        use_def.defs.extend(accesses.writes);
    };
    for (statement_index, statement) in data.statements.iter().enumerate() {
        let mut accesses = LocalAccesses::default();
        accesses.visit_statement(statement, Location { block: bb, statement_index });
        record(accesses);
    }
    let mut accesses = LocalAccesses::default();
    accesses.visit_terminator(
        data.terminator(),
        Location { block: bb, statement_index: data.statements.len() },
    );
    record(accesses);
    use_def
}

/// The locals live on entry to each block of `blocks`, whose successors are `succs`: read on
/// some path from the block's start before being assigned.
pub(crate) fn live_in(
    blocks: &[BasicBlock],
    succs: &HashMap<BasicBlock, Vec<BasicBlock>>,
    use_def: &HashMap<BasicBlock, UseDef>,
) -> HashMap<BasicBlock, HashSet<Local>> {
    let mut live_in: HashMap<BasicBlock, HashSet<Local>> =
        blocks.iter().map(|&bb| (bb, use_def[&bb].uses.clone())).collect();
    let mut changed = true;
    while changed {
        changed = false;
        // Backward problem: visiting blocks in reverse converges faster, not just correctly.
        for &bb in blocks.iter().rev() {
            let mut live_out: HashSet<Local> = HashSet::new();
            for succ in succs.get(&bb).into_iter().flatten() {
                if let Some(succ_live) = live_in.get(succ) {
                    live_out.extend(succ_live);
                }
            }
            let block_use_def = &use_def[&bb];
            let entry = live_in.get_mut(&bb).expect("block has a live-in set");
            for local in live_out {
                if !block_use_def.defs.contains(&local) && entry.insert(local) {
                    changed = true;
                }
            }
        }
    }
    live_in
}

/// Where the blocks of a function need phis, and the order to lower them in.
pub(crate) struct SsaPlan {
    /// The blocks in reverse postorder.
    pub(crate) order: Vec<BasicBlock>,
    /// Each block's immediate dominator; the entry block has none.
    pub(crate) idom: HashMap<BasicBlock, BasicBlock>,
    /// The locals that need a block argument at each block.
    pub(crate) phis: HashMap<BasicBlock, Vec<Local>>,
}

/// Plans SSA construction over `blocks` (the entry first) with successors `succs`, given the
/// locals each block assigns (`defs`) and the locals live on entry to each (`live_in`).
pub(crate) fn plan(
    blocks: &[BasicBlock],
    succs: &HashMap<BasicBlock, Vec<BasicBlock>>,
    defs: &HashMap<BasicBlock, HashSet<Local>>,
    live_in: &HashMap<BasicBlock, HashSet<Local>>,
) -> SsaPlan {
    let index: HashMap<BasicBlock, usize> =
        blocks.iter().enumerate().map(|(i, &bb)| (bb, i)).collect();
    let graph = Graph::new(
        blocks
            .iter()
            .map(|bb| {
                succs
                    .get(bb)
                    .into_iter()
                    .flatten()
                    .filter_map(|succ| index.get(succ).copied())
                    .collect()
            })
            .collect(),
    );
    let rpo = graph.reverse_postorder();
    let idom = graph.immediate_dominators(&rpo);
    let frontiers = graph.dominance_frontiers(&idom);

    let mut def_blocks: HashMap<Local, Vec<usize>> = HashMap::new();
    for (bb, locals) in defs {
        let Some(&node) = index.get(bb) else { continue };
        for &local in locals {
            def_blocks.entry(local).or_default().push(node);
        }
    }
    let mut phis: HashMap<BasicBlock, Vec<Local>> = HashMap::new();
    for (local, nodes) in def_blocks {
        for node in iterated_frontier(&frontiers, nodes) {
            let bb = blocks[node];
            if live_in.get(&bb).is_some_and(|live| live.contains(&local)) {
                phis.entry(bb).or_default().push(local);
            }
        }
    }
    // Deterministic block-argument order, whatever the hash maps iterated in.
    for locals in phis.values_mut() {
        locals.sort();
    }

    SsaPlan {
        order: rpo.iter().map(|&node| blocks[node]).collect(),
        idom: rpo[1..]
            .iter()
            .filter_map(|&node| Some((blocks[node], blocks[idom[node]?])))
            .collect(),
        phis,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// entry → header ⇄ body, header → exit: a loop.
    fn while_loop() -> Graph {
        Graph::new(vec![vec![1], vec![2, 3], vec![1], vec![]])
    }

    /// entry → then | else → join: a diamond.
    fn diamond() -> Graph {
        Graph::new(vec![vec![1, 2], vec![3], vec![3], vec![]])
    }

    #[test]
    fn reverse_postorder_puts_dominators_first() {
        let rpo = while_loop().reverse_postorder();
        assert_eq!(rpo[0], 0);
        let pos = |n| rpo.iter().position(|&x| x == n).unwrap();
        assert!(pos(1) < pos(2) && pos(1) < pos(3));
    }

    #[test]
    fn reverse_postorder_skips_unreachable_nodes() {
        let graph = Graph::new(vec![vec![1], vec![], vec![1]]);
        assert_eq!(graph.reverse_postorder(), [0, 1]);
    }

    #[test]
    fn immediate_dominators_of_a_diamond() {
        let graph = diamond();
        let idom = graph.immediate_dominators(&graph.reverse_postorder());
        assert_eq!(idom, [Some(0), Some(0), Some(0), Some(0)]);
    }

    #[test]
    fn immediate_dominators_of_a_loop() {
        let graph = while_loop();
        let idom = graph.immediate_dominators(&graph.reverse_postorder());
        assert_eq!(idom, [Some(0), Some(0), Some(1), Some(1)]);
    }

    #[test]
    fn the_loop_header_is_in_the_body_frontier() {
        let graph = while_loop();
        let frontiers =
            graph.dominance_frontiers(&graph.immediate_dominators(&graph.reverse_postorder()));
        assert_eq!(frontiers[2], HashSet::from([1]));
        assert_eq!(frontiers[1], HashSet::from([1]));
        assert!(frontiers[0].is_empty());
    }

    #[test]
    fn a_diamond_needs_a_phi_only_at_the_join() {
        let graph = diamond();
        let frontiers =
            graph.dominance_frontiers(&graph.immediate_dominators(&graph.reverse_postorder()));
        assert_eq!(iterated_frontier(&frontiers, [1]), HashSet::from([3]));
        assert!(iterated_frontier(&frontiers, [0]).is_empty());
    }

    #[test]
    fn nested_loops_need_phis_at_both_headers() {
        // 0 → 1 (outer header) → 2 (inner header) ⇄ 3 (inner body); 2 → 4 (outer latch) → 1;
        // 1 → 5 (exit). A variable assigned in the inner body meets its other values at both
        // headers.
        let graph = Graph::new(vec![vec![1], vec![2, 5], vec![3, 4], vec![2], vec![1], vec![]]);
        let frontiers =
            graph.dominance_frontiers(&graph.immediate_dominators(&graph.reverse_postorder()));
        assert_eq!(iterated_frontier(&frontiers, [0, 3]), HashSet::from([1, 2]));
    }

    #[test]
    fn phis_are_pruned_to_live_locals() {
        let bb = BasicBlock::from_usize;
        let blocks = [bb(0), bb(1), bb(2), bb(3)];
        let succs = HashMap::from([
            (bb(0), vec![bb(1), bb(2)]),
            (bb(1), vec![bb(3)]),
            (bb(2), vec![bb(3)]),
        ]);
        let (live, dead) = (Local::from_usize(1), Local::from_usize(2));
        let defs = HashMap::from([
            (bb(1), HashSet::from([live, dead])),
            (bb(2), HashSet::from([live, dead])),
        ]);
        let live_in = HashMap::from([(bb(3), HashSet::from([live]))]);
        let plan = plan(&blocks, &succs, &defs, &live_in);
        assert_eq!(plan.phis, HashMap::from([(bb(3), vec![live])]));
        assert_eq!(plan.idom.get(&bb(3)), Some(&bb(0)));
        assert_eq!(plan.order[0], bb(0));
    }
}
