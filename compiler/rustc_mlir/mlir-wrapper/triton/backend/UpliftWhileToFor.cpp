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

#include "UpliftWhileToFor.h"

#include <optional>

#include "llvm/ADT/STLExtras.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/SCF/Transforms/Transforms.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {
namespace triton {

namespace {

/// The index of the `scf.while` result that carries the induction variable of
/// a loop `scf::upliftWhileToForLoop` could uplift: the `before` block
/// argument compared by its single `arith.cmpi`, as forwarded to its
/// `scf.condition`.
std::optional<unsigned> inductionVarResult(scf::WhileOp loop) {
  Block *before = loop.getBeforeBody();
  if (!llvm::hasSingleElement(before->without_terminator()))
    return std::nullopt;
  auto cmp = dyn_cast<arith::CmpIOp>(before->front());
  if (!cmp)
    return std::nullopt;
  for (Value operand : cmp->getOperands()) {
    auto arg = dyn_cast<BlockArgument>(operand);
    if (!arg || arg.getOwner() != before)
      continue;
    for (auto [index, forwarded] :
         llvm::enumerate(loop.getConditionOp().getArgs()))
      if (forwarded == arg)
        return index;
  }
  return std::nullopt;
}

/// Uplifts `loop` to `scf.for` if it is counted, like
/// `scf::populateUpliftWhileToForPatterns`, correcting the value the induction
/// variable has after the loop.
///
/// `scf::upliftWhileToForLoop` replaces that `scf.while` result with
/// `lb + (ceildiv(ub - lb, step) - 1) * step`, the induction variable of the
/// last iteration. The `scf.while` result is the value that failed the
/// condition, `lb + max(ceildiv(ub - lb, step), 0) * step`, so a kernel that
/// reads its counter after the loop (teeny-kernels' maxpool2d) read it one step
/// short, or below `lb` when the loop didn't run at all.
void upliftWhileToFor(RewriterBase &rewriter, scf::WhileOp loop) {
  // The uplift rewrites these uses in place to its own exit value, so they
  // are collected first and redirected to the correct one afterwards.
  SmallVector<OpOperand *> exitUses;
  if (std::optional<unsigned> result = inductionVarResult(loop))
    for (OpOperand &use : loop.getResult(*result).getUses())
      exitUses.push_back(&use);

  rewriter.setInsertionPoint(loop);
  FailureOr<scf::ForOp> forOp = scf::upliftWhileToForLoop(rewriter, loop);
  if (failed(forOp) || exitUses.empty())
    return;

  rewriter.setInsertionPointAfter(*forOp);
  Location loc = forOp->getLoc();
  Value lb = forOp->getLowerBound();
  Value ub = forOp->getUpperBound();
  Value step = forOp->getStep();
  Type type = step.getType();
  auto constant = [&](int64_t value) -> Value {
    if (isa<IndexType>(type))
      return arith::ConstantIndexOp::create(rewriter, loc, value);
    return arith::ConstantIntOp::create(rewriter, loc, type, value);
  };
  Value stepDec = arith::SubIOp::create(rewriter, loc, step, constant(1));
  Value len = arith::SubIOp::create(rewriter, loc, ub, lb);
  len = arith::AddIOp::create(rewriter, loc, len, stepDec);
  Value trips = arith::DivSIOp::create(rewriter, loc, len, step);
  trips = arith::MaxSIOp::create(rewriter, loc, trips, constant(0));
  Value advance = arith::MulIOp::create(rewriter, loc, trips, step);
  Value exit = arith::AddIOp::create(rewriter, loc, lb, advance);
  for (OpOperand *use : exitUses)
    rewriter.modifyOpInPlace(use->getOwner(), [&] { use->set(exit); });
}

} // namespace

void upliftWhileLoopsToFor(ModuleOp module) {
  // Post-order, so inner loops are uplifted before the loops around them;
  // the uplift erases only the loop it rewrites.
  SmallVector<scf::WhileOp> loops;
  module.walk([&](scf::WhileOp loop) { loops.push_back(loop); });
  IRRewriter rewriter(module.getContext());
  for (scf::WhileOp loop : loops)
    upliftWhileToFor(rewriter, loop);
}

} // namespace triton
} // namespace mlir
