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

use std::collections::{HashMap, HashSet};
use std::rc::Rc;

use melior::ir::attribute::{FloatAttribute, IntegerAttribute};
use melior::ir::operation::{OperationBuilder, OperationLike};
use melior::ir::r#type::IntegerType;
use melior::ir::{
    Block, BlockLike, BlockRef, Location, Operation, RegionLike, TypeLike, Value, ValueLike,
};
use melior::utility::register_all_llvm_translations;
use rustc_abi::{FieldIdx, FieldsShape, Size as AbiSize};
use rustc_ast::{FloatTy, IntTy, UintTy};
use rustc_index::IndexVec;
use rustc_middle::mir::interpret::{CtfeProvenance, GlobalAlloc, Scalar, alloc_range};
use rustc_middle::mono::MonoItem;
use rustc_middle::mir::{
    AggregateKind, BasicBlock, BasicBlockData, BinOp, Body, CastKind, Const, ConstOperand,
    ConstValue, Local, NonDivergingIntrinsic, Operand, Place, ProjectionElem, Rvalue, Statement,
    StatementKind, UnOp, UnevaluatedConst as MirUnevaluatedConst,
};
use rustc_middle::ty::consts::ConstExt;
use rustc_middle::ty::layout::MaybeResult;
use rustc_middle::ty::{
    self, AdtDef, ConstKind, EarlyBinder, GenericArg, Instance, ParamConst, ScalarInt, Ty, TyCtxt,
    TyKind, TypingEnv, adjustment::PointerCoercion,
};
use rustc_mlir::load_all_dialects;
use rustc_mlir::shared::arith::{
    Int, Predicate, create_constant, create_int_constant, create_negf, create_subi,
};
use rustc_mlir::shared::attr::create_scalar_attr;
use rustc_mlir::shared::builtin::{tensor_type, tensor_type_like};
use rustc_mlir::shared::ub::create_ub_poison;
use rustc_mlir::triton::tensor::splat;
use rustc_mlir::triton::{create_func, int_to_ptr, load_triton_dialect};
use rustc_span::{DUMMY_SP, Span};

use crate::mlir::MlirModule;
use crate::mlir::codegen::Codegen;
use crate::mlir::codegen::triton::location::span_to_location;
use crate::mlir::codegen::triton::types::TypeMapper;
use crate::mlir::errors::MlirError;

mod location;
mod ops;
mod ssa;
mod types;

use types::is_option_ty;

type SsaValues<'c, 'p> = HashMap<Local, Value<'c, 'p>>;

/// The value of an `Option<T>` local. These locals are never inserted into `SsaValues`.
#[derive(Clone, Copy)]
pub(crate) enum OptionValue<'c, 'p> {
    /// Statically `None`.
    None,
    /// Statically `Some`, holding this value.
    Some(Value<'c, 'p>),
    /// `Some` or `None` depending on the path taken to a join: the merge of the variants that
    /// reach it, as block arguments (see `branch_operands`). `payload` is meaningful only where
    /// `is_some` (an `i1`) holds.
    Merged { is_some: Value<'c, 'p>, payload: Value<'c, 'p> },
}

impl<'c, 'p> From<Option<Value<'c, 'p>>> for OptionValue<'c, 'p> {
    fn from(value: Option<Value<'c, 'p>>) -> Self {
        value.map_or(OptionValue::None, OptionValue::Some)
    }
}

/// Tracks `Option<T>` MIR locals.
type OptionTable<'c, 'p> = LocalTable<OptionValue<'c, 'p>>;

/// Tracks tuple MIR locals (including the return place `_0` for tuple-returning functions).
/// The vec contains the individual MLIR values for each tuple field in order.
type TupleTable<'c, 'p> = LocalTable<Vec<Value<'c, 'p>>>;

/// Constant integer arrays (e.g. shape arrays like `[BLOCK_SIZE]`).
/// Keyed by the MIR `Local` that holds the array; value is the elements as `i64`.
type ConstArrays = LocalTable<Vec<i64>>;

/// Maps a raw-pointer local (from `&raw const arr`) to the `ConstArrays` key it was derived from.
type PtrToConstArray = LocalTable<Local>;

/// Maps a MIR local that holds `&[i32]` (built from a const array via `slice_from_raw_parts` /
/// `transmute`) to the shape extracted from the underlying `ConstArrays` entry.
type SliceShape = LocalTable<Vec<i64>>;

/// Runtime (non-constant) arrays: elements are SSA `Value`s instead of `i64` literals.
type DynArrays<'c, 'p> = LocalTable<Vec<Value<'c, 'p>>>;

/// Maps a raw-pointer local derived from a `dyn_array` to the source array local.
type PtrToDynArray = LocalTable<Local>;

/// Slice (fat pointer) whose elements are runtime SSA values — parallel to `SliceShape` for
/// the dynamic case.
type SliceDynValues<'c, 'p> = LocalTable<Vec<Value<'c, 'p>>>;

/// A side table for MIR locals whose values have no single SSA form (Options, tuples, constant
/// and runtime arrays, slices, block pointers, descriptor block shapes), one entry per local.
///
/// Options and tuples are merged at joins like `ssa_values` (see `branch_operands`). The other
/// tables can't be merged at all. Recording a local that is assigned in more than one place it
/// can't be merged from is an error, rather than silently keeping one assignment's value.
pub(crate) struct LocalTable<V> {
    entries: HashMap<Local, V>,
    multiply_assigned: Rc<MultiplyAssigned>,
    /// What the table holds, for the error message.
    kind: &'static str,
}

impl<V> LocalTable<V> {
    fn new(kind: &'static str) -> Self {
        Self { entries: HashMap::new(), multiply_assigned: Rc::default(), kind }
    }

    fn set_multiply_assigned(&mut self, multiply_assigned: &Rc<MultiplyAssigned>) {
        self.multiply_assigned = Rc::clone(multiply_assigned);
    }

    pub(crate) fn get(&self, local: &Local) -> Option<&V> {
        self.entries.get(local)
    }

    fn entries(&self) -> &HashMap<Local, V> {
        &self.entries
    }

    fn replace_entries(&mut self, entries: HashMap<Local, V>) {
        self.entries = entries;
    }

    pub(crate) fn contains_key(&self, local: &Local) -> bool {
        self.entries.contains_key(local)
    }

    pub(crate) fn insert(&mut self, local: Local, value: V) -> Result<(), MlirError> {
        if let Some(spans) = self.multiply_assigned.get(&local) {
            let places: Vec<String> = spans.iter().map(|span| format!("{span:?}")).collect();
            return Err(MlirError::CodegenFailed {
                err: format!(
                    "{local:?} is assigned in more than one place ({}), but its value ({}) \
                     can't be merged where they meet; assign it once",
                    places.join(", "),
                    self.kind
                ),
            });
        }
        self.entries.insert(local, value);
        Ok(())
    }
}

/// Locals with more than one definition, mapped to the source span of each definition.
type MultiplyAssigned = HashMap<Local, Vec<Span>>;

/// Locals with more than one definition, mapped to the block and source span of each.
type Definitions = HashMap<Local, Vec<(BasicBlock, Span)>>;

/// Locals with more than one whole-local definition (assignment or call destination) in the
/// `reachable` blocks of `mir`. Blocks pruned by constant folding don't count, so a definition in
/// a dead branch doesn't make a local ambiguous.
fn multiply_assigned_locals(mir: &Body<'_>, reachable: &HashSet<BasicBlock>) -> Definitions {
    let mut definitions: Definitions = HashMap::new();
    for (bb, block) in mir.basic_blocks.iter_enumerated() {
        if !reachable.contains(&bb) {
            continue;
        }
        let assigned = block.statements.iter().filter_map(|statement| match &statement.kind {
            StatementKind::Assign(assign) => Some((assign.0, statement.source_info.span)),
            _ => None,
        });
        let called = block.terminator.as_ref().and_then(|terminator| match &terminator.kind {
            rustc_middle::mir::TerminatorKind::Call { destination, .. } => {
                Some((*destination, terminator.source_info.span))
            }
            _ => None,
        });
        for (place, span) in assigned.chain(called) {
            if place.projection.is_empty() {
                definitions.entry(place.local).or_default().push((bb, span));
            }
        }
    }
    definitions.retain(|_, spans| spans.len() > 1);
    definitions
}

/// Maps a descriptor local to its block shape (the tile dimensions it loads).
type DescBlockShapes = LocalTable<Vec<i64>>;

/// Maps a MIR local to its statically-known discriminant value (u64).
/// Built by pre-scanning for `SetDiscriminant` and `Discriminant` assignments
/// so that SwitchInt on Option<T> locals can be constant-folded during codegen.
type ConstDiscLocals = HashMap<Local, u64>;

/// A block pointer from `make_block_ptr`. Triton 3.6 removed `tt.make_tensor_ptr` and
/// `tt.advance`, keeping block pointers purely in its frontend, so the backend does the same:
/// a block pointer is this record rather than an MLIR value, and `advance` only adds to
/// `offsets`. Integer entries are `i64`, as in Triton's frontend.
#[derive(Clone)]
pub(crate) struct BlockPtr<'c, 'p> {
    #[expect(dead_code, reason = "read when block-pointer load/store is added to the Triton API")]
    pub(crate) base: Value<'c, 'p>,
    #[expect(dead_code, reason = "read when block-pointer load/store is added to the Triton API")]
    pub(crate) shape: Vec<Value<'c, 'p>>,
    #[expect(dead_code, reason = "read when block-pointer load/store is added to the Triton API")]
    pub(crate) strides: Vec<Value<'c, 'p>>,
    pub(crate) offsets: Vec<Value<'c, 'p>>,
    #[expect(dead_code, reason = "read when block-pointer load/store is added to the Triton API")]
    pub(crate) block_shape: Vec<i64>,
    #[expect(dead_code, reason = "read when block-pointer load/store is added to the Triton API")]
    pub(crate) order: Vec<i64>,
}

/// The block arguments holding a local's merged value at a join.
#[derive(Clone)]
pub(crate) enum PhiArg<'c, 'p> {
    Value(Value<'c, 'p>),
    /// An Option: whether it is `Some` (an `i1`) and its payload.
    Option {
        is_some: Value<'c, 'p>,
        payload: Value<'c, 'p>,
    },
    /// A tuple: one argument per field.
    Tuple(Vec<Value<'c, 'p>>),
}

/// The state of the join-merged locals at some point: `ssa_values`, Options and tuples.
pub(crate) struct SsaSnapshot<'c, 'p> {
    values: SsaValues<'c, 'p>,
    options: HashMap<Local, OptionValue<'c, 'p>>,
    tuples: HashMap<Local, Vec<Value<'c, 'p>>>,
}

/// Codegen state threaded through all statement/terminator handlers.
pub(crate) struct CodegenState<'c, 'p> {
    pub(crate) ssa_values: SsaValues<'c, 'p>,
    pub(crate) option_table: OptionTable<'c, 'p>,
    pub(crate) tuple_fields: TupleTable<'c, 'p>,
    /// Constant integer arrays derived from aggregate literals.
    pub(crate) const_arrays: ConstArrays,
    /// `local → array_local`: a raw pointer derived from a const-array alloca.
    pub(crate) ptr_to_const_array: PtrToConstArray,
    /// `local → shape`: a fat-pointer slice `&[i32]` whose shape is statically known.
    pub(crate) slice_shape: SliceShape,
    /// Runtime arrays whose elements are MLIR SSA values.
    pub(crate) dyn_arrays: DynArrays<'c, 'p>,
    /// `ptr_local → array_local`: pointer derived from a `dyn_arrays` entry.
    pub(crate) ptr_to_dyn_array: PtrToDynArray,
    /// Fat-pointer slices whose elements are runtime MLIR values.
    pub(crate) slice_dyn_values: SliceDynValues<'c, 'p>,
    /// Block shapes for tensor descriptors: `desc_local → [dim0, dim1, ...]`.
    pub(crate) desc_block_shapes: DescBlockShapes,
    /// Block pointers, which have no MLIR value (see `BlockPtr`).
    pub(crate) block_ptrs: LocalTable<BlockPtr<'c, 'p>>,
    /// Statically-known discriminant values for Option<T> locals (from SetDiscriminant).
    /// Used by `codegen_switch_int` to constant-fold branches on Option discriminants.
    pub(crate) const_disc_locals: ConstDiscLocals,
    /// The MIR locals that need a block argument (a phi) at each block of the function body,
    /// from SSA construction (see `ssa`).
    pub(crate) phi_join_locals: HashMap<BasicBlock, Vec<Local>>,
    /// (block, local) → the block arguments holding that local's merged value.
    pub(crate) phi_block_args: HashMap<(BasicBlock, Local), PhiArg<'c, 'p>>,
    /// The MLIR type of the payload of each `Option<T>` local that needs a block argument,
    /// where `T` maps to one: a branch carrying `None` passes a placeholder of this type.
    pub(crate) phi_option_payload_types: HashMap<Local, melior::ir::Type<'c>>,
    /// Blocks with phis whose arguments are fixed: the first branch lowered into a block creates
    /// them (see `branch_operands`).
    pub(crate) phi_blocks_entered: HashSet<BasicBlock>,
}

impl<'c, 'p> CodegenState<'c, 'p> {
    /// The values of the locals that are merged at joins, to resume from in the blocks that a
    /// block dominates.
    fn ssa_snapshot(&self) -> SsaSnapshot<'c, 'p> {
        SsaSnapshot {
            values: self.ssa_values.clone(),
            options: self.option_table.entries().clone(),
            tuples: self.tuple_fields.entries().clone(),
        }
    }

    fn restore_ssa_snapshot(&mut self, snapshot: &SsaSnapshot<'c, 'p>) {
        self.ssa_values = snapshot.values.clone();
        self.option_table.replace_entries(snapshot.options.clone());
        self.tuple_fields.replace_entries(snapshot.tuples.clone());
    }

    /// Gives every `LocalTable` that can't merge at joins the function's multiply-assigned
    /// locals. Options and tuples are merged at joins like `ssa_values`, so they get none.
    fn set_multiply_assigned(&mut self, definitions: &Definitions) {
        let multiply_assigned: MultiplyAssigned = definitions
            .iter()
            .map(|(&local, defs)| (local, defs.iter().map(|&(_, span)| span).collect()))
            .collect();
        let multiply_assigned = &Rc::new(multiply_assigned);
        self.const_arrays.set_multiply_assigned(multiply_assigned);
        self.ptr_to_const_array.set_multiply_assigned(multiply_assigned);
        self.slice_shape.set_multiply_assigned(multiply_assigned);
        self.dyn_arrays.set_multiply_assigned(multiply_assigned);
        self.ptr_to_dyn_array.set_multiply_assigned(multiply_assigned);
        self.slice_dyn_values.set_multiply_assigned(multiply_assigned);
        self.desc_block_shapes.set_multiply_assigned(multiply_assigned);
        self.block_ptrs.set_multiply_assigned(multiply_assigned);
    }

    fn new() -> Self {
        Self {
            ssa_values: HashMap::new(),
            option_table: LocalTable::new("an Option"),
            tuple_fields: LocalTable::new("a tuple"),
            const_arrays: LocalTable::new("a constant array"),
            ptr_to_const_array: LocalTable::new("a pointer to a constant array"),
            slice_shape: LocalTable::new("a constant slice"),
            dyn_arrays: LocalTable::new("an array of runtime values"),
            ptr_to_dyn_array: LocalTable::new("a pointer to a runtime array"),
            slice_dyn_values: LocalTable::new("a slice of runtime values"),
            desc_block_shapes: LocalTable::new("a tensor descriptor"),
            block_ptrs: LocalTable::new("a block pointer"),
            const_disc_locals: HashMap::new(),
            phi_join_locals: HashMap::new(),
            phi_block_args: HashMap::new(),
            phi_option_payload_types: HashMap::new(),
            phi_blocks_entered: HashSet::new(),
        }
    }
}

/// The locals of `mir` with a statically known value, and the blocks reachable once the
/// switches on them are folded, in DFS preorder.
///
/// The two depend on each other: a definition in a block that a folded switch never reaches
/// doesn't count against a local's value (e.g. the `Some` arm of a `USE_BIAS = false` branch).
/// Starting from every block, each round drops the blocks the known values prove unreachable,
/// until nothing changes. Every round's answer holds for any execution, since an execution only
/// runs definitions in blocks the round considers reachable.
fn known_locals_and_reachable_blocks<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: &Instance<'tcx>,
    mir: &Body<'tcx>,
) -> (HashMap<Local, u64>, HashSet<BasicBlock>, Vec<BasicBlock>) {
    let mut considered: HashSet<BasicBlock> = mir.basic_blocks.indices().collect();
    loop {
        let known = compute_const_disc_locals(tcx, instance, mir, &considered);
        let (reachable, ordered) = compute_reachable_blocks(tcx, instance, mir, &known);
        if reachable == considered {
            return (known, reachable, ordered);
        }
        considered = reachable;
    }
}

/// How a whole-local definition determines the local's value, for `compute_const_disc_locals`.
enum KnownDef {
    /// A constant, or the variant set by `SetDiscriminant`.
    Const(u64),
    /// The discriminant of another local, or a copy of it.
    From(Local),
    /// Anything else: a computed value, a call result, an enum aggregate.
    Unknown,
}

/// The locals with a statically known value: a discriminant (for `Option`s set by
/// `SetDiscriminant`) or a scalar constant, following `Discriminant` and copies. A local is
/// known only if every definition of it in the `considered` blocks gives the same value, so a
/// local chosen by an `if` (`f = 1` on one arm, `f = 2` on the other) isn't folded to either.
///
/// Constants are instantiated via `instance`, so const generic parameters (e.g. `USE_BIAS`) get
/// their concrete values after monomorphization.
fn compute_const_disc_locals<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: &Instance<'tcx>,
    mir: &Body<'tcx>,
    considered: &HashSet<BasicBlock>,
) -> HashMap<Local, u64> {
    use rustc_middle::mir::interpret::Scalar;

    let extract_scalar_const = |c: &ConstOperand<'tcx>| -> Option<u64> {
        // Instantiate the constant (substitutes generic params like USE_BIAS → false).
        let instantiated = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, c.const_),
        );
        match instantiated {
            Const::Val(ConstValue::Scalar(Scalar::Int(s)), _) => Some(s.to_bits_unchecked() as u64),
            Const::Ty(_, ref ck) => {
                if let ConstKind::Value(cv) = ck.kind() {
                    cv.valtree.try_to_leaf().map(|s| s.to_bits_unchecked() as u64)
                } else {
                    None
                }
            }
            _ => None,
        }
    };

    let mut defs: HashMap<Local, Vec<KnownDef>> = HashMap::new();
    for (bb, bb_data) in mir.basic_blocks.iter_enumerated() {
        if !considered.contains(&bb) {
            continue;
        }
        for stmt in &bb_data.statements {
            let (place, def) = match &stmt.kind {
                StatementKind::SetDiscriminant { place, variant_index } => {
                    (**place, KnownDef::Const(variant_index.as_u32() as u64))
                }
                StatementKind::Assign(assign) => {
                    let (place, rvalue) = assign.as_ref();
                    let def = match rvalue {
                        Rvalue::Use(Operand::Constant(c), _) => {
                            extract_scalar_const(c).map_or(KnownDef::Unknown, KnownDef::Const)
                        }
                        Rvalue::Discriminant(src) => KnownDef::From(src.local),
                        Rvalue::Use(Operand::Copy(p) | Operand::Move(p), _)
                            if p.projection.is_empty() =>
                        {
                            KnownDef::From(p.local)
                        }
                        _ => KnownDef::Unknown,
                    };
                    (*place, def)
                }
                _ => continue,
            };
            if place.projection.is_empty() {
                defs.entry(place.local).or_default().push(def);
            }
        }
        if let rustc_middle::mir::TerminatorKind::Call { destination, .. } =
            &bb_data.terminator().kind
            && destination.projection.is_empty()
        {
            defs.entry(destination.local).or_default().push(KnownDef::Unknown);
        }
    }

    // A local becomes known once all its definitions resolve to one value; resolving one can
    // resolve the locals copied from it, so repeat until nothing changes.
    let mut result: HashMap<Local, u64> = HashMap::new();
    let mut changed = true;
    while changed {
        changed = false;
        for (local, local_defs) in &defs {
            if result.contains_key(local) {
                continue;
            }
            let mut values = local_defs.iter().map(|def| match def {
                KnownDef::Const(value) => Some(*value),
                KnownDef::From(src) => result.get(src).copied(),
                KnownDef::Unknown => None,
            });
            let Some(Some(first)) = values.next() else { continue };
            if values.all(|value| value == Some(first)) {
                result.insert(*local, first);
                changed = true;
            }
        }
    }
    result
}

/// Extract a constant integer value from a SwitchInt discriminant operand, if statically known.
///
/// Handles:
/// 1. `Operand::Constant` with a scalar integer value (e.g. `BIAS = false`)
/// 2. `Operand::Copy/Move` of a local in `const_disc_locals` (from SetDiscriminant / Discriminant)
pub(crate) fn extract_switch_const<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: &Instance<'tcx>,
    discr: &Operand<'tcx>,
    const_disc_locals: &HashMap<Local, u64>,
) -> Option<u64> {
    use rustc_middle::mir::interpret::Scalar;
    let result = match discr {
        Operand::Constant(c) => {
            // Instantiate generic const params (e.g. USE_BIAS → false) before extracting value.
            let instantiated = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(tcx, c.const_),
            );
            match instantiated {
                Const::Val(ConstValue::Scalar(Scalar::Int(s)), _) => {
                    Some(s.to_bits_unchecked() as u64)
                }
                Const::Ty(_, ref ck) => {
                    if let ConstKind::Value(cv) = ck.kind() {
                        cv.valtree.try_to_leaf().map(|s| s.to_bits_unchecked() as u64)
                    } else {
                        None
                    }
                }
                _ => None,
            }
        }
        Operand::Copy(p) | Operand::Move(p) if p.projection.is_empty() => {
            const_disc_locals.get(&p.local).copied()
        }
        _ => None,
    };
    result
}

/// BFS reachability analysis from bb0, constant-folding `SwitchInt` when the discriminant
/// is statically known.  Returns (set, vec) where the vec preserves BFS order — the correct
/// topological order for SSA value availability.  Use the vec for block *processing* order
/// and the set for O(1) membership tests.
fn compute_reachable_blocks<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: &Instance<'tcx>,
    mir: &Body<'tcx>,
    const_disc_locals: &HashMap<Local, u64>,
) -> (HashSet<BasicBlock>, Vec<BasicBlock>) {
    use rustc_middle::mir::TerminatorKind;

    let mut reachable: HashSet<BasicBlock> = HashSet::new();
    let mut ordered: Vec<BasicBlock> = Vec::new();
    let mut queue: Vec<BasicBlock> = vec![BasicBlock::from_u32(0)];

    while let Some(bb) = queue.pop() {
        if reachable.contains(&bb) {
            continue;
        }
        let bb_data = &mir.basic_blocks[bb];
        // A block whose own terminator is `unreachable` is formal-only (typically the
        // exhaustiveness-only `otherwise` arm of a `SwitchInt` over an already-fully-covered
        // discriminant, e.g. `codegen_switch_binary`'s two-explicit-cases pattern). Nothing can
        // execute it, so it's excluded from both the reachable set and codegen order entirely --
        // no MLIR block is created or emitted for it -- rather than trying to lower a body that
        // will never run.
        if matches!(bb_data.terminator().kind, TerminatorKind::Unreachable) {
            continue;
        }
        reachable.insert(bb);
        ordered.push(bb);
        queue.extend(block_successors(tcx, instance, mir, bb, const_disc_locals));
    }

    (reachable, ordered)
}

/// The blocks control can pass to from `bb`: every successor, except that a `SwitchInt` on a
/// statically known discriminant goes to the one target it selects.
fn block_successors<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: &Instance<'tcx>,
    mir: &Body<'tcx>,
    bb: BasicBlock,
    const_disc_locals: &HashMap<Local, u64>,
) -> Vec<BasicBlock> {
    use rustc_middle::mir::TerminatorKind;

    let terminator = mir.basic_blocks[bb].terminator();
    if let TerminatorKind::SwitchInt { discr, targets } = &terminator.kind
        && let Some(const_val) = extract_switch_const(tcx, instance, discr, const_disc_locals)
    {
        let target = targets
            .iter()
            .find(|(val, _)| *val == const_val as u128)
            .map(|(_, bb)| bb)
            .unwrap_or_else(|| targets.otherwise());
        return vec![target];
    }
    match &terminator.kind {
        TerminatorKind::Return | TerminatorKind::Unreachable => Vec::new(),
        // A call's unwind edge never runs: kernels are built with `panic=abort`.
        TerminatorKind::Call { target, .. } => target.iter().copied().collect(),
        TerminatorKind::Drop { target, .. } | TerminatorKind::Assert { target, .. } => {
            vec![*target]
        }
        _ => terminator.successors().collect(),
    }
}

/// SSA construction for the function's blocks (see `ssa`).
fn plan_function_ssa<'tcx>(
    tcx: TyCtxt<'tcx>,
    instance: &Instance<'tcx>,
    mir: &Body<'tcx>,
    reachable_order: &[BasicBlock],
    reachable: &HashSet<BasicBlock>,
    const_disc_locals: &HashMap<Local, u64>,
) -> ssa::SsaPlan {
    let successors: HashMap<BasicBlock, Vec<BasicBlock>> = reachable_order
        .iter()
        .map(|&bb| {
            let succs = block_successors(tcx, instance, mir, bb, const_disc_locals)
                .into_iter()
                .filter(|succ| reachable.contains(succ))
                .collect();
            (bb, succs)
        })
        .collect();
    let use_def: HashMap<BasicBlock, ssa::UseDef> =
        reachable_order.iter().map(|&bb| (bb, ssa::block_use_def(mir, bb))).collect();
    let live_in = ssa::live_in(reachable_order, &successors, &use_def);
    let defs: HashMap<BasicBlock, HashSet<Local>> =
        use_def.into_iter().map(|(bb, use_def)| (bb, use_def.defs)).collect();
    ssa::plan(reachable_order, &successors, &defs, &live_in)
}

pub(crate) struct TritonCodegen<'a> {
    module: &'a MlirModule<'static>,
    type_mapper: TypeMapper,
}

impl<'a> TritonCodegen<'a> {
    pub fn new(module: &'a MlirModule<'static>) -> Self {
        let context = module.context();

        load_all_dialects(context);
        register_all_llvm_translations(context);
        load_triton_dialect(context);

        Self { module, type_mapper: TypeMapper::new() }
    }

    fn to_scalar_int<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        node: &Operand<'tcx>,
    ) -> Result<ScalarInt, MlirError> {
        match node {
            Operand::Constant(c) => {
                // We expect the constant to have a value that tells us the discriminant/variant
                match c.const_ {
                    Const::Val(ConstValue::Scalar(Scalar::Int(scalar_int)), _) => Ok(scalar_int),
                    Const::Ty(ty, const_val) => {
                        // Handle const generic parameters of integer type
                        if !ty.is_integral() {
                            return Err(MlirError::InvalidScalar { node: format!("{:?}", node) });
                        }
                        match const_val.kind() {
                            ConstKind::Param(param) => {
                                let value = instance.args.const_at(param.index as usize).to_value();
                                let scalar_int = value.try_to_leaf().ok_or_else(|| {
                                    MlirError::InvalidScalar { node: format!("{:?}", node) }
                                })?;
                                Ok(scalar_int)
                            }
                            _ => Err(MlirError::InvalidScalar { node: format!("{:?}", node) }),
                        }
                    }
                    _ => Err(MlirError::InvalidScalar { node: format!("{:?}", node) }),
                }
            }
            _ => Err(MlirError::InvalidScalar { node: format!("{:?}", node) }),
        }
    }

    /// Evaluate an operand to an `i32` at MLIR-codegen time, performing limited
    /// constant folding on MIR locals when needed.
    ///
    /// This handles the common pattern where `HEAD_DIM / 2` (a const-generic
    /// expression) is not folded by rustc into a literal constant before being
    /// passed to a Triton operation, resulting in MIR like:
    ///
    /// ```text
    /// _12 = Div(const 16_i32, const 2_i32)   // HEAD_DIM / 2
    /// _k  = T::arange(const 0_i32, copy _12)
    /// ```
    ///
    /// `to_scalar_int` only handles `Operand::Constant`; this method also
    /// handles `Operand::Copy` by scanning the MIR body for the local's
    /// assignment and recursively evaluating it as a constant integer.
    pub(crate) fn try_eval_const_i32<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        node: &Operand<'tcx>,
    ) -> Result<i32, MlirError> {
        // Fast path: literal constant.
        if let Ok(si) = self.to_scalar_int(tcx, instance, node) {
            return Ok(si.to_i32());
        }
        // Slow path: local variable — try constant folding through the MIR.
        if let Operand::Copy(place) | Operand::Move(place) = node {
            if place.projection.is_empty() {
                if let Some(val) = Self::fold_local_as_i32(tcx, instance, mir, place.local) {
                    return Ok(val);
                }
            }
        }
        Err(MlirError::InvalidScalar { node: format!("{:?}", node) })
    }

    /// Recursively constant-fold a MIR local to an `i32` value by scanning all
    /// basic-block statements for its assignment and evaluating the rvalue.
    ///
    /// Handles `Use`, `BinaryOp`, and `UnaryOp` rvalues whose operands are
    /// themselves foldable (either literal constants or other folded locals).
    fn fold_local_as_i32<'tcx>(
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        local: Local,
    ) -> Option<i32> {
        for block in mir.basic_blocks.iter() {
            for stmt in &block.statements {
                if let StatementKind::Assign(assign) = &stmt.kind {
                    let (place, rvalue) = assign.as_ref();
                    if place.local != local || !place.projection.is_empty() {
                        continue;
                    }
                    return match rvalue {
                        Rvalue::Use(op, _) => Self::fold_operand_as_i32(tcx, instance, mir, op),
                        Rvalue::BinaryOp(bin_op, operands) => {
                            let (lhs, rhs) = operands.as_ref();
                            let l = Self::fold_operand_as_i32(tcx, instance, mir, lhs)?;
                            let r = Self::fold_operand_as_i32(tcx, instance, mir, rhs)?;
                            match bin_op {
                                BinOp::Add => Some(l.wrapping_add(r)),
                                BinOp::Sub => Some(l.wrapping_sub(r)),
                                BinOp::Mul => Some(l.wrapping_mul(r)),
                                BinOp::Div if r != 0 => Some(l.wrapping_div(r)),
                                BinOp::Rem if r != 0 => Some(l.wrapping_rem(r)),
                                BinOp::Shl => Some(l.wrapping_shl(r as u32)),
                                BinOp::Shr => Some(l.wrapping_shr(r as u32)),
                                BinOp::BitAnd => Some(l & r),
                                BinOp::BitOr => Some(l | r),
                                BinOp::BitXor => Some(l ^ r),
                                _ => None,
                            }
                        }
                        Rvalue::UnaryOp(un_op, op) => {
                            let v = Self::fold_operand_as_i32(tcx, instance, mir, op)?;
                            match un_op {
                                UnOp::Neg => Some(v.wrapping_neg()),
                                UnOp::Not => Some(!v),
                                _ => None,
                            }
                        }
                        _ => None,
                    };
                }
            }
        }
        None
    }

    /// Evaluate a single MIR operand to `i32`, delegating to `fold_local_as_i32`
    /// for `Copy`/`Move` operands.
    fn fold_operand_as_i32<'tcx>(
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        op: &Operand<'tcx>,
    ) -> Option<i32> {
        match op {
            Operand::Constant(c) => match c.const_ {
                Const::Val(ConstValue::Scalar(Scalar::Int(si)), _) => Some(si.to_i32()),
                Const::Ty(ty, const_val) if ty.is_integral() => {
                    if let ConstKind::Param(param) = const_val.kind() {
                        let value = instance.args.const_at(param.index as usize).to_value();
                        Some(value.try_to_leaf()?.to_i32())
                    } else {
                        None
                    }
                }
                _ => None,
            },
            Operand::Copy(p) | Operand::Move(p) if p.projection.is_empty() => {
                Self::fold_local_as_i32(tcx, instance, mir, p.local)
            }
            _ => None,
        }
    }

    /// Try to extract i64 values from a promoted constant of type `&[i32; N]`.
    ///
    /// This handles the MIR pattern generated by `&[BLOCK_SIZE]` when coercing to `&[i32]`:
    ///   `_x = const promoted[N] as &[i32] (PointerCoercion(Unsize, Implicit))`
    /// The promoted constant is a reference to a static allocation containing the array elements.
    /// Returns `Some(vec_of_i64_values)` on success, `None` if the pattern doesn't match.
    fn try_read_array_ref_const<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: Instance<'tcx>,
        src_ty: Ty<'tcx>,
        c: &Const<'tcx>,
    ) -> Option<Vec<i64>> {
        // Source type must be &[i32; N] (reference to fixed-size integral array).
        let (elem_ty, len_const) = match src_ty.kind() {
            TyKind::Ref(_, inner, _) => match inner.kind() {
                TyKind::Array(et, lc) => (et, lc),
                _ => return None,
            },
            _ => return None,
        };
        if !elem_ty.is_integral() {
            return None;
        }
        let elem_size_bytes = match elem_ty.kind() {
            TyKind::Int(IntTy::I8) | TyKind::Uint(UintTy::U8) => 1u64,
            TyKind::Int(IntTy::I16) | TyKind::Uint(UintTy::U16) => 2,
            TyKind::Int(IntTy::I32) | TyKind::Uint(UintTy::U32) => 4,
            TyKind::Int(IntTy::I64) | TyKind::Uint(UintTy::U64) => 8,
            _ => return None,
        };
        let n = len_const.try_to_target_usize(tcx)? as usize;

        // Resolve the constant to a ConstValue.
        // Promoted constants from generic functions arrive as Const::Unevaluated with generic args;
        // we must substitute the instance's args then evaluate via const_eval_resolve.
        let const_val: ConstValue = match c {
            Const::Val(cv, _) => *cv,
            Const::Unevaluated(uv, _) => {
                // Substitute the function's generic params with the instance's concrete args.
                let concrete_args =
                    EarlyBinder::bind(tcx, uv.args).instantiate(tcx, instance.args).skip_norm_wip();
                let concrete_uv =
                    MirUnevaluatedConst { def: uv.def, args: concrete_args, promoted: uv.promoted };
                match tcx.const_eval_resolve(
                    TypingEnv::fully_monomorphized(),
                    concrete_uv,
                    DUMMY_SP,
                ) {
                    Ok(cv) => cv,
                    Err(_) => return None,
                }
            }
            _ => return None,
        };

        // The ConstValue for `&[T; N]` is a thin pointer (Scalar::Ptr) to the allocation.
        // Dereference it to get the allocation that holds the array bytes.
        let (alloc_id, base_offset) = match const_val {
            ConstValue::Scalar(Scalar::Ptr(ptr, _)) => {
                let (prov, offset): (CtfeProvenance, _) = ptr.prov_and_relative_offset();
                (prov.alloc_id(), offset)
            }
            ConstValue::Indirect { alloc_id, offset } => (alloc_id, offset),
            _ => return None,
        };

        let GlobalAlloc::Memory(const_alloc) = tcx.global_alloc(alloc_id) else { return None };
        let alloc = const_alloc.inner();

        let mut result = Vec::with_capacity(n);
        for i in 0..n {
            let elem_offset = base_offset + AbiSize::from_bytes(i as u64 * elem_size_bytes);
            let range = alloc_range(elem_offset, AbiSize::from_bytes(elem_size_bytes));
            let scalar = alloc.read_scalar(&tcx, range, false).ok()?;
            let val = match scalar {
                Scalar::Int(s) => match elem_size_bytes {
                    1 => s.to_i8() as i64,
                    2 => s.to_i16() as i64,
                    4 => s.to_i32() as i64,
                    8 => s.to_i64(),
                    _ => return None,
                },
                _ => return None,
            };
            result.push(val);
        }
        Some(result)
    }

    /// The value is assumed to be a scalar of the same type as the tensor.
    /// The result is a tensor of the same shape as the provided tensor, with the scalar values repeated.
    fn like_tensor<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        location: Location<'a>,
        tensor: Value<'a, 'a>,
        value: Value<'a, 'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let tensor_type = tensor_type_like(
            tensor
                .r#type()
                .try_into()
                .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?,
            value.r#type(),
        )
        .map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;

        let splat_op: Operation<'_> =
            splat(&self.module.context(), location, value, tensor_type.into())
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let result = splat_op.result(0).unwrap();

        mlir_block.append_operation(splat_op);
        Ok(result.into())
    }

    fn codegen_function<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        fn_ty: Ty<'tcx>,
        instance: &Instance<'tcx>,
    ) -> Result<(), MlirError> {
        let mut state = CodegenState::new();
        let mut basic_blocks: HashMap<BasicBlock, BlockRef<'_, '_>> = HashMap::new();

        // Downcast to a FnSig
        let fn_sig = fn_ty.fn_sig(tcx);
        let fn_sig = fn_sig.skip_binder(); // Remove late-bound lifetimes

        // Extract a friendly function name, preferring unmangled if possible
        let func_name = tcx.symbol_name(*instance).name;

        // Try to demangle using the Rust symbol demangling crate if available.
        // Since in rustc we don't always bring in the rustc-demangle crate, but
        // the symbol_name should generally be readable for non-generic items.
        // Otherwise, fallback to `def_path_str` (should give a crate-relative path).
        let _friendly_name = if func_name.starts_with("_R") {
            // Looks like a Rust-mangled symbol. Try to show a better name.
            tcx.def_path_str(instance.def_id())
        } else {
            func_name.to_string()
        };

        // Skip Triton intrinsic function bodies — calls to these are intercepted at call-sites
        // by the codegen dispatch table, so the actual body is never compiled for GPU execution.
        // Their signatures may contain types (like `&[T]`) that have no valid static MLIR form.
        if fn_sig.inputs().iter().any(|ty| {
            matches!(ty.kind(), TyKind::Ref(_, inner, _) if matches!(inner.kind(), TyKind::Slice(_)))
        }) {
            return Ok(());
        }

        // Arguments — Option<T> params are excluded from the MLIR signature; they
        // are tracked in the option_table as None (absent) by default.
        let inputs: Vec<Ty<'tcx>> = fn_sig.inputs().iter().copied().collect();
        let arg_types: Vec<_> = inputs
            .iter()
            .filter(|ty| !is_option_ty(tcx, **ty))
            .map(|ty| self.type_mapper.map_type(self.module.context(), &tcx, ty))
            .collect();

        // Result type — flatten top-level Rust tuples into multiple MLIR return types.
        let ret_type = fn_sig.output();
        let ret_types: Vec<_> = if ret_type.is_unit() {
            vec![]
        } else if let TyKind::Tuple(elem_tys) = ret_type.kind() {
            elem_tys
                .iter()
                .map(|ty| self.type_mapper.map_type(self.module.context(), &tcx, &ty))
                .collect()
        } else {
            match self.type_mapper.map_type(self.module.context(), &tcx, &ret_type).to_result() {
                Ok(t) => vec![t],
                Err(_) => vec![],
            }
        };

        // Skip functions whose MLIR signature contains dynamic tensor types (tensor<?x...>).
        // These are Triton intrinsic stubs whose call-sites are intercepted by the dispatch table;
        // their generic tensor parameters have no statically-known shape and cannot be verified.
        let has_dynamic_tensor = arg_types.iter().chain(ret_types.iter()).any(|ty| ty.is_tensor());
        if has_dynamic_tensor {
            return Ok(());
        }

        // Iterate over MIR basic blocks and codegen each one
        let visibility = if func_name.ends_with("entry_point") { "public" } else { "private" };

        let func_loc =
            span_to_location(self.module.context(), tcx, tcx.def_span(instance.def_id()));
        let func_op: Operation<'_> = create_func(
            self.module.context(),
            func_loc,
            func_name,
            visibility,
            &arg_types,
            &ret_types,
            16,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();

        let mir = tcx.instance_mir(instance.def);
        let location = func_loc;


        if func_name.contains("program_id") {
            //println!("[DEBUG] TritonCodegen::codegen_function: mir: {:?}", mir);
        }

        // Statically known discriminants and constants (e.g. Options set to None by const
        // generics) fold switches; only the blocks still reachable get MLIR blocks.
        let (const_disc_locals, reachable_bbs, bfs_ordered_bbs) =
            known_locals_and_reachable_blocks(tcx, instance, mir);
        state.const_disc_locals = const_disc_locals.clone();

        // Loops are lowered as plain branches like any other control flow; the Triton backend
        // lifts them to `scf` (Backend::structureControlFlow, teenyc-3a1).
        state.set_multiply_assigned(&multiply_assigned_locals(mir, &reachable_bbs));

        let ssa_plan = plan_function_ssa(
            tcx,
            instance,
            mir,
            &bfs_ordered_bbs,
            &reachable_bbs,
            &const_disc_locals,
        );
        state.phi_join_locals = ssa_plan.phis.clone();
        state.phi_option_payload_types = self.option_payload_types(tcx, instance, mir, &ssa_plan);

        // Create MLIR blocks for all reachable MIR blocks.
        // Index order is fine for block *creation* — MLIR blocks just need to exist before
        // they are branched to; actual codegen (where SSA values must be available) uses BFS order.
        for bb in &bfs_ordered_bbs {
            let bb = *bb;

            let block = Block::new(&[]);
            if bb.index() == 0 {
                // Add non-Option function arguments as block arguments to the entry block.
                // Option<T> params are pre-populated into option_table as None.
                let mut mlir_arg_idx = 0;
                for (param_idx, input_ty) in inputs.iter().enumerate() {
                    let local = Local::from_usize(param_idx + 1);
                    if is_option_ty(tcx, *input_ty) {
                        state.option_table.insert(local, OptionValue::None)?;
                    } else {
                        let value = block.add_argument(arg_types[mlir_arg_idx], location);
                        state.ssa_values.insert(local, value);
                        mlir_arg_idx += 1;
                    }
                }
            }

            // Phi block args for join blocks are created lazily in codegen_goto /
            // codegen_switch_int so that tensor locals get their concrete shape type
            // (known only at codegen time) rather than the generic tensor<?xf32>.

            let block_ref =
                func_op.region(0).expect("tt.func must have a body region").append_block(block);
            basic_blocks.insert(bb, block_ref);
        }

        // Lower the blocks in reverse postorder, each starting from the values at the end of
        // its immediate dominator, plus its block arguments. Every value a block reads is then
        // defined in a block that dominates it, whatever path reached it.
        let mut values_at_end: HashMap<BasicBlock, SsaSnapshot<'_, '_>> = HashMap::new();
        for &bb in &ssa_plan.order {
            let bb_data = &mir.basic_blocks[bb];
            if let Some(idom) = ssa_plan.idom.get(&bb) {
                state.restore_ssa_snapshot(&values_at_end[idom]);
            }
            for local in ssa_plan.phis.get(&bb).into_iter().flatten() {
                match state.phi_block_args.get(&(bb, *local)).cloned() {
                    Some(PhiArg::Value(arg)) => {
                        state.ssa_values.insert(*local, arg);
                    }
                    Some(PhiArg::Option { is_some, payload }) => {
                        state
                            .option_table
                            .insert(*local, OptionValue::Merged { is_some, payload })?;
                    }
                    Some(PhiArg::Tuple(fields)) => state.tuple_fields.insert(*local, fields)?,
                    None => {}
                }
            }

            self.codegen_basic_block(tcx, instance, mir, bb, bb_data, &mut state, &basic_blocks)?;
            values_at_end.insert(bb, state.ssa_snapshot());
        }

        self.module.mlir.body().append_operation(func_op.into());

        Ok(())
    }

    /// The MLIR payload type of each `Option<T>` local that needs a block argument, where `T`
    /// maps to a scalar: a branch carrying `None` into the join passes a placeholder of it.
    /// (A tensor's MIR type maps to no concrete shape, so a `None` tensor Option can only be
    /// merged once a `Some` has fixed the arguments.)
    fn option_payload_types<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        plan: &ssa::SsaPlan,
    ) -> HashMap<Local, melior::ir::Type<'a>> {
        let mut types = HashMap::new();
        for &local in plan.phis.values().flatten() {
            let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(tcx, mir.local_decls[local].ty),
            );
            if !is_option_ty(tcx, ty) {
                continue;
            }
            let TyKind::Adt(_, args) = ty.kind() else { continue };
            let payload_ty = args.type_at(0);
            if matches!(
                payload_ty.kind(),
                TyKind::Int(_) | TyKind::Uint(_) | TyKind::Float(_) | TyKind::Bool
            ) {
                types.insert(
                    local,
                    self.type_mapper.map_type(self.module.context(), &tcx, &payload_ty),
                );
            }
        }
        types
    }

    fn codegen_basic_block<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        bb: BasicBlock,
        bb_data: &BasicBlockData<'tcx>,
        state: &mut CodegenState<'a, 'a>,
        basic_blocks: &HashMap<BasicBlock, BlockRef<'a, 'a>>,
    ) -> Result<(), MlirError> {
        let mlir_block = basic_blocks.get(&bb).expect("block not found");

        // Codegen each MIR statement in order.
        for stmt in &bb_data.statements {
            self.codegen_statement(tcx, instance, mir, stmt, mlir_block, state)?;
        }

        // Codegen the block terminator.
        self.codegen_terminator(
            tcx,
            instance,
            mir,
            bb_data.terminator(),
            mlir_block,
            state,
            basic_blocks,
        )?;

        Ok(())
    }

    fn codegen_statement<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        stmt: &Statement<'tcx>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        //println!("[DEBUG] TritonCodegen::codegen_statement: ssa_values: {:?}", state.ssa_values);
        let location = span_to_location(self.module.context(), tcx, stmt.source_info.span);
        match &stmt.kind {
            StatementKind::Assign(assign) => {
                let (place, rvalue) = assign.as_ref();
                //println!("[DEBUG] TritonCodegen::codegen_statement: Assign: {:?}, {:?} {:?}", stmt, place, rvalue);
                self.codegen_assign(tcx, instance, mir, place, rvalue, location, mlir_block, state)
            }
            StatementKind::SetDiscriminant { place, variant_index } => self
                .codegen_set_discriminant(
                    tcx,
                    instance,
                    mir,
                    place,
                    *variant_index,
                    mlir_block,
                    state,
                ),
            StatementKind::StorageLive(local) => self.codegen_storage_live(tcx, *local, mlir_block),
            StatementKind::StorageDead(local) => self.codegen_storage_dead(tcx, *local, mlir_block),
            StatementKind::Intrinsic(intrinsic) => {
                self.codegen_intrinsic(tcx, intrinsic, mlir_block)
            }
            // Runtime no-ops or analysis-only statements that require no codegen.
            StatementKind::Nop
            | StatementKind::ConstEvalCounter
            | StatementKind::FakeRead(_)
            | StatementKind::PlaceMention(_)
            | StatementKind::AscribeUserType(..)
            | StatementKind::Coverage(_)
            | StatementKind::BackwardIncompatibleDropHint { .. } => Ok(()),
        }?;

        //println!("[DEBUG] TritonCodegen::codegen_statement: ssa_values: {:?}", state.ssa_values);
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn codegen_assign<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        place: &Place<'tcx>,
        rvalue: &Rvalue<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        match rvalue {
            Rvalue::Use(operand, _) => {
                let ty = operand.ty(mir, tcx);
                let typing_env = TypingEnv::fully_monomorphized();
                let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    typing_env,
                    EarlyBinder::bind(tcx, ty),
                );

                // Copies of a fn pointer carry no value either (see the ReifyFnPointer cast);
                // `resolve_fn_ptr_instance` follows them back to the fn item.
                if matches!(normalized_ty.kind(), TyKind::FnPtr(..)) {
                    return Ok(());
                }

                // Route Option<T> locals into the option_table instead of ssa_values.
                if is_option_ty(tcx, normalized_ty) {
                    let opt_value = match operand {
                        Operand::Copy(src) | Operand::Move(src) => {
                            *state.option_table.get(&src.local).unwrap_or_else(|| {
                                panic!("Option local {:?} not found in option_table", src.local)
                            })
                        }
                        // `None::<T>` as a constant — no inner value.
                        Operand::Constant(_) => OptionValue::None,
                        Operand::RuntimeChecks(_) => {
                            todo!("RuntimeChecks operand not yet supported")
                        }
                    };
                    state.option_table.insert(place.local, opt_value)?;
                    return Ok(());
                }

                // Route Tuple<T...> locals into tuple_fields.
                if let TyKind::Tuple(elem_tys) = normalized_ty.kind() {
                    let fields = match operand {
                        Operand::Copy(src) | Operand::Move(src) => {
                            state.tuple_fields.get(&src.local).cloned().unwrap_or_else(|| {
                                panic!("Tuple local {:?} not found in tuple_fields", src.local)
                            })
                        }
                        Operand::Constant(const_op) => self.codegen_tuple_constant(
                            tcx, instance, const_op, elem_tys, location, mlir_block,
                        )?,
                        Operand::RuntimeChecks(_) => todo!("RuntimeChecks for tuple"),
                    };
                    state.tuple_fields.insert(place.local, fields)?;
                    return Ok(());
                }

                // Propagate side-table entries for locals tracked outside ssa_values.
                // Copy/Move of an array/pointer/slice local just aliases the side-table entry.
                if let Operand::Copy(src) | Operand::Move(src) = operand {
                    if src.projection.is_empty() {
                        // dyn_array copy: _dest = copy _src where _src is in dyn_arrays.
                        if let Some(elems) = state.dyn_arrays.get(&src.local).cloned() {
                            state.dyn_arrays.insert(place.local, elems)?;
                            return Ok(());
                        }
                        // const_array copy: _dest = copy _src where _src is in const_arrays.
                        // MIR GVN produces this for a repeated shape literal (`&[n]` twice),
                        // and a constant array has no SSA value to fall back on.
                        if let Some(elems) = state.const_arrays.get(&src.local).cloned() {
                            state.const_arrays.insert(place.local, elems)?;
                            return Ok(());
                        }
                        // Block pointers have no SSA value either.
                        if let Some(bp) = state.block_ptrs.get(&src.local).cloned() {
                            state.block_ptrs.insert(place.local, bp)?;
                            return Ok(());
                        }
                        // const_array pointer copy.
                        if let Some(&arr_local) = state.ptr_to_const_array.get(&src.local) {
                            state.ptr_to_const_array.insert(place.local, arr_local)?;
                            return Ok(());
                        }
                        // dyn_array pointer copy.
                        if let Some(&arr_local) = state.ptr_to_dyn_array.get(&src.local) {
                            state.ptr_to_dyn_array.insert(place.local, arr_local)?;
                            return Ok(());
                        }
                        // slice_dyn_values copy.
                        if let Some(vals) = state.slice_dyn_values.get(&src.local).cloned() {
                            state.slice_dyn_values.insert(place.local, vals)?;
                            return Ok(());
                        }
                        // slice_shape copy.
                        if let Some(shape) = state.slice_shape.get(&src.local).cloned() {
                            state.slice_shape.insert(place.local, shape)?;
                            return Ok(());
                        }
                        // ADT/Range locals stored in tuple_fields (e.g. Range moves for for-loops).
                        if let Some(fields) = state.tuple_fields.get(&src.local).cloned() {
                            state.tuple_fields.insert(place.local, fields)?;
                            return Ok(());
                        }
                    }
                }

                // `_x = const &[T; N]` — populate slice_shape so PointerCoercion(Unsize)
                // can propagate the shape when this local is later coerced to &[T].
                if let Operand::Constant(c) = operand {
                    let const_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                        tcx,
                        TypingEnv::fully_monomorphized(),
                        EarlyBinder::bind(tcx, c.const_.ty()),
                    );
                    if let Some(shape) =
                        self.try_read_array_ref_const(tcx, *instance, const_ty, &c.const_)
                    {
                        state.slice_shape.insert(place.local, shape)?;
                        // Fall through to codegen_operand to emit a placeholder SSA value.
                    }
                }

                let result = self.codegen_operand(
                    tcx,
                    instance,
                    operand,
                    normalized_ty,
                    location,
                    mlir_block,
                    state,
                )?;
                state.ssa_values.insert(place.local, result);
            }
            Rvalue::Cast(cast_kind, operand, ty) => {
                // A fn item reified to a fn pointer is only ever a combine function handed to
                // `reduce` / `associative_scan`, which resolve the target from the MIR (see
                // `resolve_fn_ptr_instance`) and generate its body inline in their combine
                // region. Triton has no runtime function pointers, so no value is emitted.
                if matches!(cast_kind, CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer(_), _))
                {
                    return Ok(());
                }

                // PtrToPtr cast on a const-array pointer — propagate the side-table entry
                // without emitting an MLIR op (const arrays have no MLIR value).
                if matches!(cast_kind, CastKind::PtrToPtr) {
                    if let Operand::Copy(src_place) | Operand::Move(src_place) = operand {
                        if let Some(&arr_local) = state.ptr_to_const_array.get(&src_place.local) {
                            state.ptr_to_const_array.insert(place.local, arr_local)?;
                            return Ok(());
                        }
                    }
                }

                // Transmute of a slice-shape tuple ((*const T, usize) → &[T]) — propagate shape.
                if matches!(cast_kind, CastKind::Transmute) {
                    if let Operand::Copy(src_place) | Operand::Move(src_place) = operand {
                        if let Some(shape) = state.slice_shape.get(&src_place.local).cloned() {
                            state.slice_shape.insert(place.local, shape)?;
                            return Ok(());
                        }
                    }
                }

                // PointerCoercion(Unsize) from a promoted &[T; N] constant → &[T].
                // e.g. `zeros::<D>(&[BLOCK_SIZE])` generates:
                //   _23 = const promoted[41] as &[i32] (PointerCoercion(Unsize, Implicit))
                // Extract the array values and populate slice_shape so downstream calls
                // (zeros, broadcast_to, reshape, etc.) can read the tensor shape.
                if let CastKind::PointerCoercion(PointerCoercion::Unsize, _) = cast_kind {
                    if let Operand::Constant(const_op) = operand {
                        let src_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                            tcx,
                            TypingEnv::fully_monomorphized(),
                            EarlyBinder::bind(tcx, const_op.const_.ty()),
                        );
                        if let Some(shape) =
                            self.try_read_array_ref_const(tcx, *instance, src_ty, &const_op.const_)
                        {
                            state.slice_shape.insert(place.local, shape)?;
                            // Fall through to codegen_cast to emit the i64 placeholder value.
                        }
                    } else if let Operand::Copy(p) | Operand::Move(p) = operand {
                        if p.projection.is_empty() {
                            // Dynamic pointer (from a dyn_array via Ref) → slice_dyn_values.
                            if let Some(&arr_local) = state.ptr_to_dyn_array.get(&p.local) {
                                if let Some(vals) = state.dyn_arrays.get(&arr_local).cloned() {
                                    state.slice_dyn_values.insert(place.local, vals)?;
                                } else {
                                }
                            }
                            // Static pointer (from a const_array via Ref) → slice_shape.
                            if let Some(&arr_local) = state.ptr_to_const_array.get(&p.local) {
                                if let Some(shape) = state.const_arrays.get(&arr_local).cloned() {
                                    state.slice_shape.insert(place.local, shape)?;
                                }
                            }
                            // Const array ref assigned directly (e.g. `_x = const &[T; N]`) → slice_shape.
                            if let Some(shape) = state.slice_shape.get(&p.local).cloned() {
                                state.slice_shape.insert(place.local, shape)?;
                            }
                        }
                    }
                }

                let result = self.codegen_cast(
                    tcx, instance, mir, cast_kind, operand, ty, location, mlir_block, state,
                )?;

                state.ssa_values.insert(place.local, result);
            }
            Rvalue::Aggregate(aggregate_kind, index_vec) => {
                // Route Option<T> aggregates (Some/None) into the option_table.
                if let AggregateKind::Adt(def_id, variant_index, _, _, _) = aggregate_kind.as_ref()
                {
                    let adt_def = tcx.adt_def(*def_id);
                    let norm_place_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                        tcx,
                        TypingEnv::fully_monomorphized(),
                        EarlyBinder::bind(tcx, place.ty(mir, tcx).ty),
                    );
                    if is_option_ty(tcx, norm_place_ty) {
                        return self.codegen_option_aggregate(
                            tcx,
                            instance,
                            mir,
                            place,
                            adt_def,
                            *variant_index,
                            index_vec,
                            location,
                            mlir_block,
                            state,
                        );
                    }
                }

                // Route Tuple aggregates into tuple_fields (MLIR uses multiple return values).
                if let AggregateKind::Tuple = aggregate_kind.as_ref() {
                    // Special case: `(*const T, usize)` fat-pointer tuple where the pointer
                    // comes from a const array.  Route to `slice_shape` instead — no MLIR value.
                    if let Some(first_op) = index_vec.iter().next() {
                        if let Operand::Copy(p) | Operand::Move(p) = first_op {
                            if let Some(&arr_local) = state.ptr_to_const_array.get(&p.local) {
                                if let Some(shape) = state.const_arrays.get(&arr_local) {
                                    state.slice_shape.insert(place.local, shape.clone())?;
                                    return Ok(());
                                }
                            }
                            if let Some(&arr_local) = state.ptr_to_dyn_array.get(&p.local) {
                                if let Some(vals) = state.dyn_arrays.get(&arr_local).cloned() {
                                    state.slice_dyn_values.insert(place.local, vals)?;
                                    return Ok(());
                                }
                            }
                        }
                    }

                    let fields = index_vec
                        .iter()
                        .map(|op| match op {
                            Operand::Copy(p) | Operand::Move(p) => self.codegen_copy(&p, state),
                            _ => todo!("Tuple aggregate with non-copy/move operand: {:?}", op),
                        })
                        .collect::<Result<Vec<_>, _>>()?;
                    state.tuple_fields.insert(place.local, fields)?;
                    return Ok(());
                }

                // Constant integer arrays (e.g. shape arrays `[BLOCK_SIZE]` for zeros/reshape).
                // We only track them as constant metadata; no MLIR value is emitted.
                if let AggregateKind::Array(_elem_ty) = aggregate_kind.as_ref() {
                    let elems: Option<Vec<i64>> = index_vec
                        .iter()
                        .map(|op| {
                            // First try literal-constant extraction, then constant-fold
                            // computed locals (e.g. `HEAD_DIM / 2` where HEAD_DIM is a
                            // const generic — the division is not folded by rustc before
                            // being used as an array-size argument like `&[HEAD_DIM / 2]`).
                            self.to_scalar_int(tcx, instance, op)
                                .ok()
                                .map(|s| match s.size().bytes() {
                                    1 => s.to_u8() as i64,
                                    2 => s.to_i16() as i64,
                                    4 => s.to_i32() as i64,
                                    8 => s.to_i64(),
                                    n => todo!("ScalarInt size {} bytes", n),
                                })
                                .or_else(|| {
                                    Self::fold_operand_as_i32(tcx, instance, mir, op)
                                        .map(|v| v as i64)
                                })
                        })
                        .collect();
                    if let Some(elems) = elems {
                        state.const_arrays.insert(place.local, elems)?;
                        return Ok(());
                    }
                    // Non-constant elements: look up each as an SSA value.
                    let dyn_elems: Result<Vec<Value<'_, '_>>, MlirError> = index_vec
                        .iter()
                        .map(|op| match op {
                            Operand::Copy(p) | Operand::Move(p) => self.codegen_copy(p, state),
                            Operand::Constant(_) | Operand::RuntimeChecks(_) => {
                                let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                                    tcx,
                                    TypingEnv::fully_monomorphized(),
                                    EarlyBinder::bind(tcx, op.ty(mir, tcx)),
                                );
                                self.codegen_operand(
                                    tcx, instance, op, ty, location, mlir_block, state,
                                )
                            }
                        })
                        .collect();
                    let dyn_elems = dyn_elems?;
                    state.dyn_arrays.insert(place.local, dyn_elems)?;
                    return Ok(());
                }

                // Generic ADT structs (e.g. core::ops::Range, custom structs): treat fields as
                // a "tuple" stored in tuple_fields so that field-projection reads work.
                // Known Triton types are handled inside codegen_aggregate_create.
                if let AggregateKind::Adt(def_id, _, _, _, _) = aggregate_kind.as_ref() {
                    let adt_def = tcx.adt_def(*def_id);
                    let adt_name = format!("{:?}", adt_def);
                    let is_triton_type = adt_name == "triton::llvm::triton::tensor::LlvmTensor"
                        || adt_name == "triton::llvm::triton::pointer::LlvmPointer";
                    if !is_triton_type {
                        let fields: Result<Vec<Value<'_, '_>>, MlirError> = index_vec
                            .iter()
                            .map(|op| match op {
                                Operand::Copy(p) | Operand::Move(p) => self.codegen_copy(p, state),
                                Operand::Constant(_) | Operand::RuntimeChecks(_) => {
                                    let ty = instance
                                        .instantiate_mir_and_normalize_erasing_regions(
                                            tcx,
                                            TypingEnv::fully_monomorphized(),
                                            EarlyBinder::bind(tcx, op.ty(mir, tcx)),
                                        );
                                    self.codegen_operand(
                                        tcx, instance, op, ty, location, mlir_block, state,
                                    )
                                }
                            })
                            .collect();
                        state.tuple_fields.insert(place.local, fields?)?;
                        return Ok(());
                    }
                }

                let result = self.codegen_aggregate_create(
                    tcx,
                    instance,
                    mir,
                    aggregate_kind,
                    index_vec,
                    location,
                    mlir_block,
                    state,
                )?;

                if let Some(result) = result {
                    state.ssa_values.insert(place.local, result);
                } else {
                }
            }
            Rvalue::Repeat(operand, _) => todo!("Repeat: {:?}", operand),
            Rvalue::Ref(_region, _borrow_kind, src_place) => {
                // Treat `&arr` the same as `&raw const arr` for const/dyn arrays so that
                // downstream slice-coercion and `make_tensor_descriptor` can recover the values.
                if src_place.projection.is_empty() {
                    if state.const_arrays.contains_key(&src_place.local) {
                        state.ptr_to_const_array.insert(place.local, src_place.local)?;
                        return Ok(());
                    }
                    if state.dyn_arrays.contains_key(&src_place.local) {
                        state.ptr_to_dyn_array.insert(place.local, src_place.local)?;
                        return Ok(());
                    }
                }
                todo!("Ref: {:?} {:?} {:?}", _region, _borrow_kind, src_place)
            }
            Rvalue::ThreadLocalRef(def_id) => todo!("ThreadLocalRef: {:?}", def_id),
            Rvalue::RawPtr(_raw_ptr_kind, src_place) => {
                // If this is `&raw const const_array_local`, record the mapping so that
                // downstream `zeros` / `slice_from_raw_parts` calls can recover the shape.
                if src_place.projection.is_empty() {
                    if state.const_arrays.contains_key(&src_place.local) {
                        state.ptr_to_const_array.insert(place.local, src_place.local)?;
                        return Ok(());
                    }
                    if state.dyn_arrays.contains_key(&src_place.local) {
                        state.ptr_to_dyn_array.insert(place.local, src_place.local)?;
                        return Ok(());
                    }
                }
                todo!("RawPtr: {:?} {:?}", _raw_ptr_kind, src_place)
            }
            Rvalue::BinaryOp(bin_op, operands) => {
                // WithOverflow variants return (T, bool) — treat as plain arithmetic + false flag.
                let is_overflow_op = matches!(
                    bin_op,
                    BinOp::AddWithOverflow | BinOp::SubWithOverflow | BinOp::MulWithOverflow
                );
                if is_overflow_op {
                    let plain_op = match bin_op {
                        BinOp::AddWithOverflow => BinOp::Add,
                        BinOp::SubWithOverflow => BinOp::Sub,
                        BinOp::MulWithOverflow => BinOp::Mul,
                        _ => unreachable!(),
                    };
                    let value = self.codegen_binary_op(
                        tcx, instance, mir, place, &plain_op, operands, location, mlir_block, state,
                    )?;
                    if let Some(result_val) = value {
                        let ctx = self.module.context();
                        let i1_ty = IntegerType::new(ctx, 1);
                        let false_attr = IntegerAttribute::new(i1_ty.into(), 0);
                        let false_op: Operation<'a> =
                            melior::dialect::arith::constant(ctx, false_attr.into(), location)
                                .into();
                        let false_val: Value<'_, '_> = false_op.result(0).unwrap().into();
                        mlir_block.append_operation(false_op);
                        state.tuple_fields.insert(place.local, vec![result_val, false_val])?;
                    }
                } else {
                    let value = self.codegen_binary_op(
                        tcx, instance, mir, place, bin_op, operands, location, mlir_block, state,
                    )?;
                    if let Some(value) = value {
                        state.ssa_values.insert(place.local, value);
                    }
                }
            }
            Rvalue::UnaryOp(un_op, operand) => {
                let op_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(tcx, operand.ty(mir, tcx)),
                );
                let val = self
                    .codegen_operand(tcx, instance, operand, op_ty, location, mlir_block, state)?;
                let result = match un_op {
                    UnOp::Neg => {
                        let val_ty = val.r#type();
                        if val_ty.is_integer() {
                            // integer negation: 0 - x
                            let zero_op: Operation<'a> = self.zero_constant(location, val_ty);
                            let zero: Value<'a, 'a> = zero_op.result(0).unwrap().into();
                            mlir_block.append_operation(zero_op);
                            let sub_op: Operation<'a> =
                                create_subi(self.module.context(), location, zero, val)
                                    .map_err(|e| MlirError::CreateOperation { err: e })?;
                            let r = sub_op.result(0).unwrap().into();
                            mlir_block.append_operation(sub_op);
                            r
                        } else {
                            let neg_op: Operation<'a> =
                                create_negf(self.module.context(), location, val)
                                    .map_err(|e| MlirError::CreateOperation { err: e })?;
                            let r = neg_op.result(0).unwrap().into();
                            mlir_block.append_operation(neg_op);
                            r
                        }
                    }
                    UnOp::Not => {
                        todo!("UnaryOp::Not: {:?}", operand)
                    }
                    UnOp::PtrMetadata => {
                        todo!("UnaryOp::PtrMetadata: {:?}", operand)
                    }
                };
                state.ssa_values.insert(place.local, result);
            }
            Rvalue::Discriminant(src_place) => {
                let src_ty = src_place.ty(mir, tcx).ty;
                let norm_src_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(tcx, src_ty),
                );
                if is_option_ty(tcx, norm_src_ty) {
                    let discr: i64 = match state.option_table.get(&src_place.local) {
                        Some(OptionValue::None) => 0,
                        Some(OptionValue::Some(_)) => 1,
                        // Chosen at run time: the discriminant is the merged `is_some` flag.
                        Some(&OptionValue::Merged { is_some, .. }) => {
                            let i8_ty = IntegerType::new(self.module.context(), 8).into();
                            let op: Operation<'a> =
                                melior::dialect::arith::extui(is_some, i8_ty, location).into();
                            let result = op
                                .result(0)
                                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
                            mlir_block.append_operation(op);
                            state.ssa_values.insert(place.local, result.into());
                            return Ok(());
                        }
                        None => panic!(
                            "Option local {:?} not found in option_table for Discriminant",
                            src_place.local
                        ),
                    };
                    let int_val = Int::I8(discr as u8);
                    let const_op: Operation<'a> =
                        create_int_constant(self.module.context(), location, int_val)
                            .map_err(|e| MlirError::CreateOperation { err: e })?
                            .into();
                    let result = const_op
                        .result(0)
                        .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
                    mlir_block.append_operation(const_op);
                    state.ssa_values.insert(place.local, result.into());
                    return Ok(());
                }
                todo!("Discriminant for non-Option: {:?}", src_place)
            }
            Rvalue::CopyForDeref(place) => todo!("CopyForDeref: {:?}", place),
            Rvalue::WrapUnsafeBinder(operand, ty) => {
                todo!("WrapUnsafeBinder: {:?} {:?}", operand, ty)
            }
            Rvalue::Reborrow(ty, mutability, place) => {
                todo!("Reborrow: {:?} {:?} {:?}", ty, mutability, place)
            }
        }

        // todo!("[TODO] TritonCodegen::codegen_assign: {:?} {:?}", place, rvalue)
        Ok(())
    }

    fn codegen_aggregate_create<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        aggregate_kind: &AggregateKind<'tcx>,
        index_vec: &IndexVec<FieldIdx, Operand<'tcx>>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        match aggregate_kind {
            AggregateKind::Adt(def_id, _, _raw_list, _, _) => {
                // Get the ADT definition and a human-readable name for debugging.
                let adt_def = tcx.adt_def(*def_id);
                let adt_name = format!("{:?}", adt_def);

                if "triton::llvm::triton::tensor::LlvmTensor" == adt_name {
                    self.codegen_create_tensor(
                        tcx, instance, mir, index_vec, location, mlir_block, state,
                    )
                } else if "triton::llvm::triton::pointer::LlvmPointer" == adt_name {
                    self.codegen_create_pointer(
                        tcx, instance, mir, index_vec, location, mlir_block, state,
                    )
                } else {
                    todo!(
                        "codegen_aggregate_create: {:?} {:?} {:?}",
                        adt_name,
                        adt_def,
                        index_vec.as_slice()
                    );
                }
            }
            _ => todo!("AggregateKind: {:?}", aggregate_kind),
        }
    }

    fn codegen_create_tensor<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        index_vec: &IndexVec<FieldIdx, Operand<'tcx>>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        _state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let arg1 = index_vec.get(FieldIdx::from_usize(0)).expect("arg1 not found");
        let arg1_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, arg1.ty(mir, tcx)),
        );

        let pointee_ty = match arg1_ty.kind() {
            TyKind::RawPtr(pointee_ty, _) => {
                self.type_mapper.map_type(self.module.context(), &tcx, &pointee_ty)
            }
            _ => todo!("codegen_create_tensor: arg1_ty: {:?}", arg1_ty),
        };
        let tensor_type = tensor_type(&[i64::MIN], pointee_ty).into();
        let tensor_op = create_ub_poison(self.module.context(), location, tensor_type)
            .map_err(|e| MlirError::CreateOperation { err: e })?;

        let tensor_result = tensor_op.result(0).unwrap();
        mlir_block.append_operation(tensor_op);
        Ok(Some(tensor_result.into()))
    }

    fn codegen_create_pointer<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        index_vec: &IndexVec<FieldIdx, Operand<'tcx>>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let arg1 = index_vec.get(FieldIdx::from_usize(0)).expect("arg1 not found");
        let arg1_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, arg1.ty(mir, tcx)),
        );
        // `Pointer<T>` is a newtype wrapper around a raw pointer in Triton DSL.
        // Preserve the wrapped pointer SSA value rather than materializing poison.
        let pointer_value =
            self.codegen_operand(tcx, instance, arg1, arg1_ty, location, mlir_block, state)?;
        Ok(Some(pointer_value))
    }

    fn codegen_const_adt<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        adt_def: &AdtDef<'tcx>,
        value: Value<'a, 'a>,
        raw_list: &[GenericArg<'tcx>],
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let name = format!("{:?}", adt_def);
        let map_ty = |idx: usize| {
            let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(tcx, raw_list[idx].expect_ty()),
            );
            self.type_mapper.map_type(self.module.context(), &tcx, &ty)
        };

        // If the name of the ADT is tensor, then we create a poison operation.
        // This is because the tensor creation is part of the dsl dead code which
        // will be eliminated by the optimizer.
        if name == "triton::llvm::triton::tensor::Tensor" {
            let ty = map_ty(0);
            let tensor_type = tensor_type(&[i64::MIN], ty).into();
            let tensor_op = create_ub_poison(self.module.context(), location, tensor_type)
                .map_err(|e| MlirError::CreateOperation { err: e })?;

            let tensor_result = tensor_op.result(0).unwrap();
            mlir_block.append_operation(tensor_op);
            Ok(tensor_result.into())
        } else if name == "triton::llvm::triton::pointer::Pointer" {
            // `Pointer<T>` is a newtype wrapper — pass through the wrapped value directly.
            Ok(value)
        } else if adt_def.is_enum() {
            // For enum ADTs (e.g. `Axis`, `PaddingOption`, `CacheModifier`, etc.),
            // `codegen_scalar_const_value` already emitted the discriminant as an integer.
            // Pass through the value unchanged.
            Ok(value)
        } else {
            todo!("Adt: {:?}", adt_def)
        }
    }

    fn codegen_binary_op<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _place: &Place<'tcx>,
        bin_op: &BinOp,
        operands: &(Operand<'tcx>, Operand<'tcx>),
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let (lhs_op, rhs_op) = operands;

        let lhs_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, lhs_op.ty(mir, tcx)),
        );
        let rhs_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, rhs_op.ty(mir, tcx)),
        );
        let lhs =
            self.codegen_operand(tcx, instance, lhs_op, lhs_ty, location, mlir_block, state)?;
        let rhs =
            self.codegen_operand(tcx, instance, rhs_op, rhs_ty, location, mlir_block, state)?;

        // `lhs_ty` is already monomorphized, so this only reads its element type.
        let unsigned = Self::is_unsigned(tcx, instance, lhs_ty);
        let cmp = |pred: Predicate| pred.for_signedness(unsigned);
        match bin_op {
            BinOp::Add | BinOp::AddUnchecked => {
                self.codegen_add(tcx, location, lhs, rhs, mlir_block)
            }
            BinOp::AddWithOverflow => todo!("BinOp::AddWithOverflow"),
            BinOp::Sub | BinOp::SubUnchecked => {
                self.codegen_sub(tcx, location, lhs, rhs, mlir_block)
            }
            BinOp::SubWithOverflow => todo!("BinOp::SubWithOverflow"),
            BinOp::Mul | BinOp::MulUnchecked => {
                self.codegen_mul(tcx, location, lhs, rhs, mlir_block)
            }
            BinOp::MulWithOverflow => todo!("BinOp::MulWithOverflow"),
            BinOp::Div => self.codegen_div(tcx, unsigned, location, lhs, rhs, mlir_block),
            BinOp::Rem => self.codegen_rem(tcx, unsigned, location, lhs, rhs, mlir_block),
            BinOp::BitXor => self.codegen_xor(tcx, location, lhs, rhs, mlir_block),
            BinOp::BitAnd => self.codegen_and(tcx, location, lhs, rhs, mlir_block),
            BinOp::BitOr => self.codegen_or(tcx, location, lhs, rhs, mlir_block),
            BinOp::Shl | BinOp::ShlUnchecked => {
                self.codegen_shl(tcx, location, lhs, rhs, mlir_block)
            }
            BinOp::Shr | BinOp::ShrUnchecked => {
                self.codegen_shr(tcx, !unsigned, location, lhs, rhs, mlir_block)
            }
            BinOp::Eq => self.codegen_cmpi(tcx, Predicate::EQ, location, lhs, rhs, mlir_block),
            BinOp::Lt => self.codegen_cmpi(tcx, cmp(Predicate::SLT), location, lhs, rhs, mlir_block),
            BinOp::Le => self.codegen_cmpi(tcx, cmp(Predicate::SLE), location, lhs, rhs, mlir_block),
            BinOp::Ne => self.codegen_cmpi(tcx, Predicate::NE, location, lhs, rhs, mlir_block),
            BinOp::Ge => self.codegen_cmpi(tcx, cmp(Predicate::SGE), location, lhs, rhs, mlir_block),
            BinOp::Gt => self.codegen_cmpi(tcx, cmp(Predicate::SGT), location, lhs, rhs, mlir_block),
            BinOp::Cmp => todo!("BinOp::Cmp"),
            BinOp::Offset => todo!("BinOp::Offset"),
        }
    }

    /// Codegen construction of an `Option<T>` aggregate (either `None` or `Some(inner)`).
    /// The result is stored in `state.option_table` rather than `state.ssa_values`.
    #[allow(clippy::too_many_arguments)]
    fn codegen_option_aggregate<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        place: &Place<'tcx>,
        _adt_def: rustc_middle::ty::AdtDef<'tcx>,
        variant_index: rustc_abi::VariantIdx,
        index_vec: &rustc_index::IndexVec<FieldIdx, Operand<'tcx>>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        if variant_index.as_usize() == 0 {
            state.option_table.insert(place.local, OptionValue::None)?;
        } else {
            // Some(inner) — codegen the single field and stash it.
            let inner_op = index_vec
                .get(FieldIdx::from_usize(0))
                .expect("Option::Some aggregate must have exactly one field");
            let inner_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(tcx, inner_op.ty(mir, tcx)),
            );
            let inner_value = self
                .codegen_operand(tcx, instance, inner_op, inner_ty, location, mlir_block, state)?;
            state.option_table.insert(place.local, OptionValue::Some(inner_value))?;
        }
        Ok(())
    }

    fn codegen_set_discriminant<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        place: &Place<'tcx>,
        variant_index: rustc_abi::VariantIdx,
        _mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        let place_ty = place.ty(mir, tcx).ty;
        let norm_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, place_ty),
        );

        if is_option_ty(tcx, norm_ty) {
            if variant_index.as_usize() == 0 {
                // Two-phase None: discriminant set to 0.
                state.option_table.insert(place.local, OptionValue::None)?;
            } else {
                // Two-phase Some: inner value was written into ssa_values[place.local] by
                // a preceding field-assignment via Place::Downcast projection.  Promote it.
                let inner = state.ssa_values.remove(&place.local).unwrap_or_else(|| {
                    panic!(
                        "SetDiscriminant Some: expected inner value in ssa_values for {:?}",
                        place.local
                    )
                });
                state.option_table.insert(place.local, OptionValue::Some(inner))?;
            }
            return Ok(());
        }

        todo!("[TODO] TritonCodegen::codegen_set_discriminant for non-Option types")
    }

    fn codegen_storage_live<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _local: Local,
        _mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<(), MlirError> {
        // NO-OP: In the context of Triton and MLIR, storage live is a no-op.
        Ok(())
    }

    fn codegen_storage_dead<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _local: Local,
        _mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<(), MlirError> {
        // NO-OP: In the context of Triton and MLIR, storage dead is a no-op.
        Ok(())
    }

    fn codegen_intrinsic<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _intrinsic: &NonDivergingIntrinsic<'tcx>,
        _mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<(), MlirError> {
        todo!("[TODO] TritonCodegen::codegen_intrinsic")
    }

    fn codegen_cast<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        cast_kind: &CastKind,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        match cast_kind {
            CastKind::PointerWithExposedProvenance => self.codegen_pointer_with_exposed_provenance(
                tcx, instance, operand, ty, location, mlir_block, state,
            ),
            CastKind::PtrToPtr => {
                self.codegen_ptr_to_ptr(tcx, instance, operand, ty, location, mlir_block, state)
            }
            CastKind::IntToInt => {
                self.codegen_int_to_int(tcx, instance, mir, operand, ty, location, mlir_block, state)
            }
            CastKind::IntToFloat => {
                // For Copy/Move operands, codegen_operand ignores the type and returns the SSA
                // value directly, so passing the destination type as a hint is safe.
                let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(tcx, *ty),
                );
                let src_val = self.codegen_operand(
                    tcx,
                    instance,
                    operand,
                    normalized_ty,
                    location,
                    mlir_block,
                    state,
                )?;
                let dest_ty =
                    self.type_mapper.map_type(self.module.context(), &tcx, &normalized_ty);
                let op: Operation<'a> =
                    if Self::is_unsigned(tcx, instance, operand.ty(mir, tcx)) {
                        melior::dialect::arith::uitofp(src_val, dest_ty, location).into()
                    } else {
                        melior::dialect::arith::sitofp(src_val, dest_ty, location).into()
                    };
                let result = op.result(0).expect("int-to-float result").into();
                mlir_block.append_operation(op);
                Ok(result)
            }
            CastKind::FloatToInt => {
                let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(tcx, *ty),
                );
                let src_val = self.codegen_operand(
                    tcx,
                    instance,
                    operand,
                    normalized_ty,
                    location,
                    mlir_block,
                    state,
                )?;
                let dest_ty =
                    self.type_mapper.map_type(self.module.context(), &tcx, &normalized_ty);
                let op: Operation<'a> = if Self::is_unsigned(tcx, instance, normalized_ty) {
                    melior::dialect::arith::fptoui(src_val, dest_ty, location).into()
                } else {
                    melior::dialect::arith::fptosi(src_val, dest_ty, location).into()
                };
                let result = op.result(0).expect("float-to-int result").into();
                mlir_block.append_operation(op);
                Ok(result)
            }
            CastKind::FloatToFloat => {
                // Scalar float widening/narrowing (e.g. `x as f64` / `x as f32`). Mirrors the
                // tensor-cast handling in ops/triton/tensor.rs's codegen_cast_call, which already
                // does this correctly for `Triton::cast` on tensors — this arm was missing here,
                // so scalar float-to-float casts fell into the `_` catch-all below and silently
                // became `ub.poison`. That poison then flows into anything built from the cast
                // result (e.g. `Float::from_f64`, whose body is itself just `value as f32`),
                // corrupting kernel constants without any compile error. See the regression test
                // `float_to_float_scalar_cast` in test_kitchen_sink.rs.
                let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(tcx, *ty),
                );
                let src_val = self.codegen_operand(
                    tcx,
                    instance,
                    operand,
                    normalized_ty,
                    location,
                    mlir_block,
                    state,
                )?;
                let dest_ty =
                    self.type_mapper.map_type(self.module.context(), &tcx, &normalized_ty);

                if src_val.r#type() == dest_ty {
                    return Ok(src_val);
                }

                let float_bits = |ty: melior::ir::Type<'a>| -> u32 {
                    let s = ty.to_string();
                    if s.contains("f16") {
                        16
                    } else if s.contains("f32") {
                        32
                    } else if s.contains("f64") {
                        64
                    } else {
                        128
                    }
                };

                let op: Operation<'a> = if float_bits(dest_ty) < float_bits(src_val.r#type()) {
                    // Narrowing: melior's typed_unary_operations wrapper for truncf is a
                    // same-type unary, so build the op directly via OperationBuilder.
                    OperationBuilder::new("arith.truncf", location)
                        .add_operands(&[src_val])
                        .add_results(&[dest_ty])
                        .build()
                        .map_err(|e| MlirError::CreateOperation {
                            err: rustc_mlir::errors::Error::IncompatibleTypes {
                                lhs: e.to_string(),
                                rhs: String::new(),
                            },
                        })?
                        .into()
                } else {
                    // Widening.
                    melior::dialect::arith::extf(src_val, dest_ty, location).into()
                };
                let result = op.result(0).expect("float-to-float cast result").into();
                mlir_block.append_operation(op);
                Ok(result)
            }
            CastKind::PointerCoercion(PointerCoercion::Unsize, _) => {
                // `&[T; N]` → `&[T]` array-to-slice coercion. The caller (see the
                // `Rvalue::Cast` handling above) has already extracted the static/dynamic
                // shape into `slice_shape`/`slice_dyn_values` when possible and deliberately
                // falls through here anyway — this SSA value is dead (shape flows through the
                // side table instead), so a poison placeholder of the destination type is the
                // documented, intentional behavior, not a silently-swallowed edge case.
                let typing_env = TypingEnv::fully_monomorphized();
                let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    typing_env,
                    EarlyBinder::bind(tcx, *ty),
                );
                let mlir_ty =
                    self.type_mapper.map_type(self.module.context(), &tcx, &normalized_ty);
                let ub_op: Operation<'a> =
                    create_ub_poison(self.module.context(), location, mlir_ty)
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let result = ub_op.result(0).expect("ub.poison result").into();
                mlir_block.append_operation(ub_op);
                Ok(result)
            }
            _ => {
                // Every other cast kind (Transmute not resolvable via slice_shape,
                // PointerExposeProvenance, other PointerCoercion variants, FnPtrToPtr, etc.).
                //
                // This used to silently emit `ub.poison` of the destination type "as a safe
                // placeholder" — which is exactly how the missing `FloatToFloat` arm above went
                // undetected: `Float::from_f64`'s `value as f32` body (and its `x as f64` call
                // sites) got compiled to poison, register-allocated on top of an unrelated live
                // value, and produced plausible-looking-but-wrong numeric output instead of a
                // compile error. Fail loudly instead so unsupported patterns surface at compile
                // time rather than as silent miscompilation.
                Err(MlirError::CodegenFailed {
                    err: format!(
                        "codegen_cast: unhandled cast kind {cast_kind:?} to {ty:?} — no MLIR \
                         lowering implemented for this cast; add a `CastKind` arm to \
                         `codegen_cast` instead of falling back to `ub.poison`"
                    ),
                })
            }
        }
    }

    fn codegen_pointer_with_exposed_provenance<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let typing_env = TypingEnv::fully_monomorphized();
        let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            typing_env,
            EarlyBinder::bind(tcx, *ty),
        );

        self.codegen_operand(tcx, instance, operand, normalized_ty, location, mlir_block, state)
    }

    fn codegen_ptr_to_ptr<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let typing_env = TypingEnv::fully_monomorphized();
        let normalized_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            typing_env,
            EarlyBinder::bind(tcx, *ty),
        );

        self.codegen_operand(tcx, instance, operand, normalized_ty, location, mlir_block, state)
    }

    /// `x as T` between integer types (including from `bool`): truncate when narrowing, and
    /// widen by the source's signedness, as Rust does. Integers are signless, so a cast that
    /// only changes signedness is a no-op.
    fn codegen_int_to_int<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        operand: &Operand<'tcx>,
        ty: &Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let typing_env = TypingEnv::fully_monomorphized();
        let normalize =
            |ty| instance.instantiate_mir_and_normalize_erasing_regions(tcx, typing_env, ty);
        let src_ty = normalize(EarlyBinder::bind(tcx, operand.ty(mir, tcx)));
        let dest_ty = normalize(EarlyBinder::bind(tcx, *ty));
        let src_val =
            self.codegen_operand(tcx, instance, operand, src_ty, location, mlir_block, state)?;
        let dest_mlir_ty = self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty);

        let width = |ty: melior::ir::Type<'a>| {
            IntegerType::try_from(ty).map(|t| t.width()).map_err(|e| MlirError::InvalidType {
                msg: format!("int-to-int cast `{src_ty}` as `{dest_ty}`: {e}"),
            })
        };
        let (src_width, dest_width) = (width(src_val.r#type())?, width(dest_mlir_ty)?);
        let op: Operation<'a> = match src_width.cmp(&dest_width) {
            std::cmp::Ordering::Equal => return Ok(src_val),
            std::cmp::Ordering::Greater => {
                melior::dialect::arith::trunci(src_val, dest_mlir_ty, location).into()
            }
            std::cmp::Ordering::Less if Self::is_unsigned(tcx, instance, src_ty) => {
                melior::dialect::arith::extui(src_val, dest_mlir_ty, location).into()
            }
            std::cmp::Ordering::Less => {
                melior::dialect::arith::extsi(src_val, dest_mlir_ty, location).into()
            }
        };
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(result.into())
    }

    fn codegen_operand<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
        normalized_ty: Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        //println!("[DEBUG] TritonCodegen::codegen_operand: ssa_values: {:?} operand: {:?}", state.ssa_values, operand);

        // For MLIR move is the same as copy
        match operand {
            Operand::Copy(place) | Operand::Move(place) => self.codegen_copy(place, state),
            Operand::Constant(const_operand) => self.codegen_constant_cast(
                tcx,
                instance,
                const_operand,
                normalized_ty,
                location,
                mlir_block,
            ),
            Operand::RuntimeChecks(_) => todo!("RuntimeChecks operand not yet supported"),
        }
    }

    /// Resolve an operand that may be `Option<T>`.
    /// Returns the inner MLIR value if the option is `Some` (or if the operand is not an Option),
    /// or `None` if the option is absent.
    pub(crate) fn codegen_option_operand<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        operand: &Operand<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, operand.ty(mir, tcx)),
        );

        if is_option_ty(tcx, ty) {
            let inner = match operand {
                Operand::Copy(p) | Operand::Move(p) => {
                    match *state.option_table.get(&p.local).unwrap_or_else(|| {
                        panic!("Option local {:?} not found in option_table", p.local)
                    }) {
                        OptionValue::None => None,
                        OptionValue::Some(value) => Some(value),
                        // Triton's optional operands (masks, accumulators) must be present or
                        // absent when the kernel is compiled.
                        OptionValue::Merged { .. } => {
                            return Err(MlirError::CodegenFailed {
                                err: format!(
                                    "{:?} is `Some` or `None` depending on the path taken, but \
                                     this optional operand must be one or the other at compile \
                                     time",
                                    p.local
                                ),
                            });
                        }
                    }
                }
                // `None::<T>` constant — no inner value.
                Operand::Constant(_) => None,
                Operand::RuntimeChecks(_) => todo!("RuntimeChecks operand not yet supported"),
            };
            Ok(inner)
        } else {
            let value =
                self.codegen_operand(tcx, instance, operand, ty, location, mlir_block, state)?;
            Ok(Some(value))
        }
    }

    /// Decode a constant tuple into individual MLIR scalar values.
    /// Reads each element from the underlying memory allocation using layout offsets.
    fn codegen_tuple_constant<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        const_op: &ConstOperand<'tcx>,
        elem_tys: &[Ty<'tcx>],
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Vec<Value<'a, 'a>>, MlirError> {
        let normalized_const = match &const_op.const_ {
            Const::Val(const_val, _) => *const_val,
            other => todo!("codegen_tuple_constant: unexpected const kind {:?}", other),
        };

        let (alloc_id, base_offset) = match normalized_const {
            ConstValue::Indirect { alloc_id, offset } => (alloc_id, offset),
            other => todo!("codegen_tuple_constant: unexpected ConstValue {:?}", other),
        };

        let GlobalAlloc::Memory(const_alloc) = tcx.global_alloc(alloc_id) else {
            todo!("codegen_tuple_constant: non-memory alloc for {:?}", alloc_id);
        };
        let alloc = const_alloc.inner();

        // Get the layout of the full tuple to obtain per-field offsets.
        let tuple_ty = const_op.const_.ty();
        let tuple_layout = tcx
            .layout_of(TypingEnv::fully_monomorphized().as_query_input(tuple_ty))
            .map_err(|e| MlirError::CodegenFailed { err: format!("layout_of failed: {:?}", e) })?;

        let FieldsShape::Arbitrary { ref offsets, .. } = tuple_layout.fields else {
            todo!("codegen_tuple_constant: unexpected FieldsShape for tuple");
        };

        let mut values = Vec::with_capacity(elem_tys.len());
        for (field_idx, field_ty) in elem_tys.iter().enumerate() {
            let normalized_field_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(tcx, *field_ty),
            );
            let field_layout = tcx
                .layout_of(TypingEnv::fully_monomorphized().as_query_input(normalized_field_ty))
                .map_err(|e| MlirError::CodegenFailed {
                    err: format!("layout_of field failed: {:?}", e),
                })?;

            let field_offset = base_offset + offsets[FieldIdx::from_usize(field_idx)];
            let range = alloc_range(field_offset, field_layout.size);
            let scalar = alloc
                .read_scalar(&tcx, range, false)
                .map_err(|e| MlirError::CodegenFailed { err: format!("read_scalar: {:?}", e) })?;

            let value = self.codegen_scalar(normalized_field_ty, scalar, location, mlir_block)?;
            values.push(value);
        }

        Ok(values)
    }

    fn codegen_copy<'tcx>(
        &self,
        place: &Place<'tcx>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        // Handle Option downcast+field projection: `(_opt as Some).0` — extract the inner value.
        if let [ProjectionElem::Downcast(_, _), ProjectionElem::Field(_, _)]
        | [ProjectionElem::Field(_, _)] = place.projection.as_slice()
        {
            if let Some(opt_val) = state.option_table.get(&place.local) {
                return Ok(match *opt_val {
                    OptionValue::Some(value) => value,
                    // Read only where the discriminant says `Some` (MIR's downcast).
                    OptionValue::Merged { payload, .. } => payload,
                    OptionValue::None => {
                        panic!("Accessing field of None Option local {:?}", place.local)
                    }
                });
            }
        }

        debug_assert!(
            !state.option_table.contains_key(&place.local),
            "BUG: Option local {:?} used as a direct MLIR value; use codegen_option_operand instead",
            place.local
        );

        // Handle a single Field projection on a tuple local stored in tuple_fields.
        if let [ProjectionElem::Field(field_idx, _)] = place.projection.as_slice() {
            if let Some(fields) = state.tuple_fields.get(&place.local) {
                let idx = field_idx.index();
                return Ok(*fields.get(idx).unwrap_or_else(|| {
                    panic!("Tuple field {} not found for local {:?}", idx, place.local)
                }));
            }
        }

        if state.block_ptrs.contains_key(&place.local) {
            return Err(MlirError::CodegenFailed {
                err: format!(
                    "block pointer {:?} used as a value: a `make_block_ptr` result can only be \
                     passed to `advance` (block-pointer load/store is not supported yet)",
                    place.local
                ),
            });
        }

        Ok(state
            .ssa_values
            .get(&place.local)
            .copied()
            .expect(format!("Value not found for local: {:?}", place.local).as_str()))
    }

    fn codegen_constant_cast<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        const_operand: &ConstOperand<'tcx>,
        normalized_ty: Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        match const_operand.const_ {
            Const::Val(const_val, ty) => {
                let value =
                    self.codegen_const_value(tcx, instance, const_val, ty, location, mlir_block)?;

                match normalized_ty.kind() {
                    TyKind::RawPtr(_, _) => {
                        let value_ty = value.r#type();
                        debug_assert!(
                            value_ty.is_integer(),
                            "Triton supports only integer pointer casts"
                        );
                        let ptr_ty =
                            self.type_mapper.map_type(self.module.context(), &tcx, &normalized_ty);
                        let cast_op: Operation<'_> =
                            int_to_ptr(self.module.context(), location, value.into(), ptr_ty)
                                .map_err(|e| MlirError::CreateOperation { err: e })?
                                .into();

                        let result = cast_op.result(0).unwrap();
                        mlir_block.append_operation(cast_op);
                        Ok(result.into())
                    }
                    TyKind::Adt(adt_def, args) => {
                        let result = self.codegen_const_adt(
                            tcx,
                            instance,
                            adt_def,
                            value,
                            args.as_slice(),
                            location,
                            mlir_block,
                        )?;
                        Ok(result)
                    }
                    TyKind::Int(_) | TyKind::Uint(_) | TyKind::Float(_) | TyKind::Bool => {
                        // Constant already has the right primitive type — return as-is.
                        Ok(value)
                    }
                    TyKind::Ref(_, _, _) | TyKind::Str => {
                        // &str, &[T], and other reference types are represented as i64
                        // fat-pointer stand-ins in the type mapper. The value already
                        // carries the right i64 type from codegen_const_value.
                        Ok(value)
                    }
                    TyKind::FnPtr(_, _) | TyKind::FnDef(_, _) => {
                        // Function pointer types are represented as i64 — return as-is.
                        Ok(value)
                    }
                    _ => todo!("Constant cast normalized_ty: {:?}", normalized_ty),
                }
            }
            Const::Ty(ty, const_val) => match const_val.kind() {
                ConstKind::Param(param) => {
                    self.codegen_param_const(tcx, instance, ty, param, location, mlir_block)
                }
                ConstKind::Infer(_infer_const) => todo!("ConstKind::Infer"),
                ConstKind::Bound(_bound_var_index_kind, _) => todo!("ConstKind::Bound"),
                ConstKind::Placeholder(_) => todo!("ConstKind::Placeholder"),
                ConstKind::Alias(_, _alias_const) => todo!("ConstKind::Alias"),
                ConstKind::Value(_) => todo!("ConstKind::Value"),
                ConstKind::Error(_) => todo!("ConstKind::Error"),
                ConstKind::Expr(_) => todo!("ConstKind::Expr"),
            },
            _ => todo!("Const: {:?}", const_operand.const_),
        }
    }

    fn codegen_param_const<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        _ty: Ty<'tcx>,
        param: ParamConst,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let value = instance.args.const_at(param.index as usize).to_value();
        self.codegen_param_value(tcx, value, location, mlir_block)
    }

    fn codegen_param_value<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        value: ty::Value<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let scalar = value.try_to_scalar();

        if let Some(scalar) = scalar {
            self.codegen_scalar(value.ty, scalar, location, mlir_block)
        } else {
            todo!("codegen_param_value: {:?} scalar: {:?}", value, scalar);
        }
    }

    fn codegen_scalar<'tcx>(
        &self,
        ty: Ty<'tcx>,
        scalar: Scalar,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        match scalar {
            Scalar::Int(scalar_int) => {
                let scalar_attr = create_scalar_attr(self.module.context(), ty, scalar_int)
                    .map_err(|e| MlirError::CreateOperation { err: e })?;

                let op =
                    create_constant(self.module.context(), location, scalar_attr.0, scalar_attr.1)
                        .map_err(|e| MlirError::CreateOperation { err: e })?;

                let op: Operation<'_> = op.into();
                let result = op.result(0).expect("Constant operation result not found");
                mlir_block.append_operation(op);
                Ok(result.into())
            }
            Scalar::Ptr(pointer, _) => todo!("Scalar::Ptr: {:?}", pointer),
        }
    }

    fn codegen_const_value<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        const_val: ConstValue,
        ty: Ty<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        match const_val {
            ConstValue::Scalar(scalar) => {
                self.codegen_scalar_const_value(tcx, instance, ty, scalar, location, mlir_block)
            }
            ConstValue::ZeroSized => {
                // Zero-sized values (fn items, unit, zero-sized structs).
                // Emit i64(0) as a harmless placeholder — these are never read
                // by any real MLIR op in the stub-based codegen.
                let zero_op: Operation<'a> =
                    create_int_constant(self.module.context(), location, Int::I64(0))
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let result = zero_op.result(0).expect("zero const").into();
                mlir_block.append_operation(zero_op);
                Ok(result)
            }
            ConstValue::Slice { alloc_id: _, meta: _ } => {
                // String literals and other slice constants (&str, &[T]).
                // The type mapper maps &str and &[T] to i64 (fat-pointer stand-in).
                // Emit i64(0) as a placeholder; the callee (device_print etc.) is a stub.
                let zero_op: Operation<'a> =
                    create_int_constant(self.module.context(), location, Int::I64(0))
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let result = zero_op.result(0).expect("slice const placeholder").into();
                mlir_block.append_operation(zero_op);
                Ok(result)
            }
            ConstValue::Indirect { alloc_id: _, offset: _ } => {
                // Indirect constants (references to static allocations).
                // Emit i64(0) as a placeholder; these are not exercised by GPU stubs.
                let zero_op: Operation<'a> =
                    create_int_constant(self.module.context(), location, Int::I64(0))
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let result = zero_op.result(0).expect("indirect const placeholder").into();
                mlir_block.append_operation(zero_op);
                Ok(result)
            }
        }
    }

    fn codegen_scalar_const_value<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        _instance: &Instance<'tcx>,
        ty: Ty<'tcx>,
        scalar: Scalar,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        match scalar {
            Scalar::Int(scalar_int) => match ty.kind() {
                TyKind::Uint(_) | TyKind::Int(_) => {
                    let value =
                        Int::from_scalar(ty, scalar_int).map_err(|e| MlirError::InvalidScalar {
                            node: format!("Invalid scalar: {:?} {:?} {:?}", e, ty, scalar_int),
                        })?;

                    let const_op: Operation<'a> =
                        create_int_constant(self.module.context(), location, value)
                            .map_err(|e| MlirError::CreateOperation { err: e })?
                            .into();

                    let result = const_op
                        .result(0)
                        .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
                    mlir_block.append_operation(const_op);
                    Ok(result.into())
                }
                TyKind::Adt(adt_def, args) => {
                    let scalar_int = match scalar {
                        Scalar::Int(s) => s,
                        Scalar::Ptr(pointer, _) => todo!("Scalar::Ptr: {:?}", pointer),
                    };

                    // For enum ADTs (e.g. `#[repr(i32)] enum Axis { X=0, Y=1, Z=2 }`),
                    // emit the discriminant as the underlying integer directly.
                    let int_val = if adt_def.is_enum() {
                        match scalar_int.size().bytes() {
                            1 => Int::I8(scalar_int.to_u8()),
                            2 => Int::I16(scalar_int.to_u16()),
                            4 => Int::I32(scalar_int.to_u32()),
                            8 => Int::I64(scalar_int.to_u64()),
                            n => todo!("Enum scalar size {} bytes", n),
                        }
                    } else {
                        // For scalar newtype ADTs (e.g. `struct I32(pub i32)`), get the inner
                        // field's primitive type so Int::from_scalar can determine the
                        // correct MLIR integer kind.
                        let variant = adt_def.non_enum_variant();
                        let inner_ty = tcx
                            .type_of(variant.fields[FieldIdx::from_usize(0)].did)
                            .instantiate(tcx, args)
                            .skip_norm_wip();
                        Int::from_scalar(inner_ty, scalar_int).map_err(|e| {
                            MlirError::InvalidScalar {
                                node: format!(
                                    "Invalid scalar: {:?} {:?} {:?}",
                                    e, inner_ty, scalar_int
                                ),
                            }
                        })?
                    };

                    let const_op: Operation<'a> =
                        create_int_constant(self.module.context(), location, int_val)
                            .map_err(|e| MlirError::CreateOperation { err: e })?
                            .into();
                    let result = const_op.result(0).unwrap();
                    mlir_block.append_operation(const_op);
                    Ok(result.into())
                }
                TyKind::Bool => {
                    // bool → i1 constant (0 = false, 1 = true)
                    let val = scalar_int.to_u8();
                    let i1_ty = IntegerType::new(self.module.context(), 1);
                    let attr = IntegerAttribute::new(i1_ty.into(), val as i64);
                    let const_op: Operation<'a> = melior::dialect::arith::constant(
                        self.module.context(),
                        attr.into(),
                        location,
                    )
                    .into();
                    let result = const_op.result(0).unwrap();
                    mlir_block.append_operation(const_op);
                    Ok(result.into())
                }
                TyKind::Float(float_ty) => {
                    // Float constants are stored as Scalar::Int with their bit pattern.
                    let (f64_val, mlir_ty) = match float_ty {
                        FloatTy::F32 => {
                            let f_val = f32::from_bits(scalar_int.to_u32()) as f64;
                            (f_val, self.type_mapper.map_type(self.module.context(), &tcx, &ty))
                        }
                        FloatTy::F64 => {
                            let f_val = f64::from_bits(scalar_int.to_u64());
                            (f_val, self.type_mapper.map_type(self.module.context(), &tcx, &ty))
                        }
                        _ => todo!("Float type: {:?}", float_ty),
                    };
                    let float_attr = FloatAttribute::new(self.module.context(), mlir_ty, f64_val);
                    let const_op: Operation<'a> = melior::dialect::arith::constant(
                        self.module.context(),
                        float_attr.into(),
                        location,
                    )
                    .into();
                    let result = const_op.result(0).unwrap();
                    mlir_block.append_operation(const_op);
                    Ok(result.into())
                }
                _ => todo!("Scalar::Int ty: {:?} {:?}", ty.kind(), ty),
            },
            Scalar::Ptr(ptr, size) => todo!("Ptr ptr: {:?}, size: {:?}", ptr, size),
        }
    }
}

impl<'a> Codegen for TritonCodegen<'a> {
    fn codegen<'tcx>(&mut self, tcx: TyCtxt<'tcx>, item: &MonoItem<'tcx>) -> Result<(), MlirError> {
        match item {
            MonoItem::Fn(instance) => {
                let fn_ty = instance.ty(tcx, TypingEnv::fully_monomorphized());
                let is_fn_ty = matches!(
                    fn_ty.kind(),
                    rustc_middle::ty::TyKind::FnDef(..) | rustc_middle::ty::TyKind::FnPtr(_, _)
                );

                if !is_fn_ty {
                    todo!(
                        "[DEBUG] TritonCodegen: instance.ty(tcx) is not a function type: {:?}",
                        fn_ty
                    );
                }

                self.codegen_function(tcx, fn_ty, instance)
            }
            MonoItem::Static(_def_id) => {
                // TODO: Implement Triton codegen for statics
                todo!()
            }
            MonoItem::GlobalAsm(_item_id) => {
                // TODO: Implement Triton codegen for global asm
                todo!()
            }
        }
    }
}
