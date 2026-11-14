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

#include <iostream>
#include <mutex>
#include <vector>

#include "Backend.h"
#include "UpliftWhileToFor.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/Conversion/ControlFlowToSCF/ControlFlowToSCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlow.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/UB/IR/UBOps.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Dominance.h"
#include "mlir/Transforms/CFGToSCF.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir {
namespace triton {

namespace {

/// MLIR's CFG-to-SCF lifting for `tt.func`. The stock transformation
/// (ControlFlowToSCFTransformation, producing scf/cf/arith/ub ops) ends an
/// infinite loop's region with `func.return`, so it only works on `func.func`;
/// this one ends it with `tt.return` of poison values instead.
///
/// It wraps the stock transformation rather than subclassing it: LLVM is built
/// without RTTI, so that class has no typeinfo for a subclass in this
/// RTTI-enabled library to refer to. CFGToSCFInterface is header-only.
class TritonControlFlowToSCF : public CFGToSCFInterface {
public:
  bool canConvertMultiSuccessorBranchOp(Operation *op) override {
    return m_cf.canConvertMultiSuccessorBranchOp(op);
  }

  FailureOr<Operation *>
  createStructuredBranchRegionOp(OpBuilder &builder, Operation *condOp,
                                 TypeRange resultTypes,
                                 MutableArrayRef<Region> regions) override {
    return m_cf.createStructuredBranchRegionOp(builder, condOp, resultTypes,
                                               regions);
  }

  LogicalResult createStructuredBranchRegionTerminatorOp(
      Location loc, OpBuilder &builder, Operation *branchRegionOp,
      Operation *replacedControlFlowOp, ValueRange results) override {
    return m_cf.createStructuredBranchRegionTerminatorOp(
        loc, builder, branchRegionOp, replacedControlFlowOp, results);
  }

  FailureOr<Operation *>
  createStructuredDoWhileLoopOp(OpBuilder &builder, Operation *replacedOp,
                                ValueRange loopVariablesInit, Value condition,
                                ValueRange loopVariablesNextIter,
                                Region &&loopBody) override {
    return m_cf.createStructuredDoWhileLoopOp(
        builder, replacedOp, loopVariablesInit, condition,
        loopVariablesNextIter, std::move(loopBody));
  }

  Value getCFGSwitchValue(Location loc, OpBuilder &builder,
                          unsigned value) override {
    return m_cf.getCFGSwitchValue(loc, builder, value);
  }

  void createCFGSwitchOp(Location loc, OpBuilder &builder, Value flag,
                         ArrayRef<unsigned> caseValues,
                         BlockRange caseDestinations,
                         ArrayRef<ValueRange> caseArguments, Block *defaultDest,
                         ValueRange defaultArgs) override {
    m_cf.createCFGSwitchOp(loc, builder, flag, caseValues, caseDestinations,
                           caseArguments, defaultDest, defaultArgs);
  }

  Value getUndefValue(Location loc, OpBuilder &builder, Type type) override {
    return m_cf.getUndefValue(loc, builder, type);
  }

  FailureOr<Operation *> createUnreachableTerminator(Location loc,
                                                     OpBuilder &builder,
                                                     Region &region) override {
    auto func = dyn_cast<FuncOp>(region.getParentOp());
    if (!func)
      return emitError(loc, "cannot create an unreachable terminator for '")
             << region.getParentOp()->getName() << "'";
    auto results = llvm::map_to_vector(func.getResultTypes(), [&](Type type) {
      return getUndefValue(loc, builder, type);
    });
    return ReturnOp::create(builder, loc, results).getOperation();
  }

private:
  ControlFlowToSCFTransformation m_cf;
};

/// A zero of `type`, for an integer, index, float or Triton pointer type, or a
/// ranked tensor of one; null for any other type.
Value createZero(OpBuilder &builder, Location loc, Type type) {
  auto tensor = dyn_cast<RankedTensorType>(type);
  Type element = tensor ? tensor.getElementType() : type;
  if (isa<IntegerType, IndexType, FloatType>(element)) {
    TypedAttr zero = builder.getZeroAttr(element);
    if (tensor)
      zero = DenseElementsAttr::get(tensor, zero);
    return arith::ConstantOp::create(builder, loc, zero);
  }
  if (isa<PointerType>(element)) {
    Value address = arith::ConstantOp::create(builder, loc,
                                              builder.getI64IntegerAttr(0));
    Value null = IntToPtrOp::create(builder, loc, element, address);
    return tensor ? SplatOp::create(builder, loc, tensor, null).getResult()
                  : null;
  }
  return {};
}

/// Replaces each `ub.poison` in `module` with a zero of its type.
///
/// The CFG-to-SCF lifting passes poison for a value on a path where it is
/// undefined, i.e. never read. Not every backend lowers poison: TritonCPU
/// converts tensors to vectors but leaves `ub.poison` tensor-typed, so it
/// can't be translated to LLVM IR. A zero is equally unobservable there.
void replacePoisonWithZeros(ModuleOp module) {
  SmallVector<ub::PoisonOp> poisons;
  module.walk([&](ub::PoisonOp poison) { poisons.push_back(poison); });
  for (ub::PoisonOp poison : poisons) {
    OpBuilder builder(poison);
    if (Value zero = createZero(builder, poison.getLoc(), poison.getType())) {
      poison.replaceAllUsesWith(zero);
      poison.erase();
    }
  }
}

} // namespace

/// Replace every `cf.cond_br %c, ^bb, ^bb` with `cf.br ^bb` in the module.
///
/// Triton's makeTTIR canonicalization can merge two identical successor blocks
/// into one, creating degenerate `cf.cond_br %c, ^bb, ^bb` operations.  The
/// standard MLIR canonicalization pass is expected to fold these, but in
/// practice it sometimes misses them when the degenerate form is created late
/// in the same pass run.  This explicit pass ensures the fold always happens
/// before makeTTGIR, which crashes on degenerate cond_brs.
static void foldDegenerateCondBranches(ModuleOp module) {
  std::vector<cf::CondBranchOp> toFold;
  module.walk([&](cf::CondBranchOp condBr) {
    if (condBr.getTrueDest() == condBr.getFalseDest()) {
      auto trueOps = condBr.getTrueDestOperands();
      auto falseOps = condBr.getFalseDestOperands();
      if (trueOps.size() == falseOps.size() &&
          std::equal(trueOps.begin(), trueOps.end(), falseOps.begin())) {
        toFold.push_back(condBr);
      }
    }
  });
  for (auto condBr : toFold) {
    OpBuilder builder(condBr);
    builder.create<cf::BranchOp>(condBr.getLoc(), condBr.getTrueDest(),
                                 condBr.getTrueDestOperands());
    condBr.erase();
  }
}

Backend::Backend(std::string target) : m_target(target) {
  // nop
}

Backend::~Backend() {
  // nop
}

void Backend::initializeLLVMTargets() {
  static std::once_flag initialized;
  std::call_once(initialized, [] {
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmParsers();
    llvm::InitializeAllAsmPrinters();
  });
}

LogicalResult Backend::structureControlFlow(MLIRContext &context,
                                            ModuleOp module) {
  // The lifting creates scf, cf (intermediate switches), arith (switch
  // values) and ub (poison) operations.
  context.loadDialect<scf::SCFDialect, cf::ControlFlowDialect,
                      arith::ArithDialect, ub::UBDialect>();

  TritonControlFlowToSCF transformation;
  bool failedToLift = false;
  module.walk([&](FuncOp func) {
    if (func.getBody().empty())
      return WalkResult::advance();
    DominanceInfo dominance(func);
    // Innermost regions first, as LiftControlFlowToSCF does, so a region's
    // structured ops are already in place when its parent is lifted.
    auto lift = [&](Operation *op) {
      for (Region &region : op->getRegions()) {
        if (failed(transformCFGToSCF(region, transformation, dominance)))
          return WalkResult::interrupt();
      }
      return WalkResult::advance();
    };
    if (func->walk<WalkOrder::PostOrder>(lift).wasInterrupted()) {
      failedToLift = true;
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  if (failedToLift) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Failed to structure control flow";
    return failure();
  }

  // The lifted loops are do-while `scf.while`s carrying their condition;
  // canonicalization reduces counted ones to the form that the uplift
  // recognises (a single compare in `before`, the induction variable advanced
  // by a loop-invariant step).
  PassManager pm(&context);
  pm.addPass(createCanonicalizerPass());
  if (failed(pm.run(module.getOperation())))
    return failure();

  upliftWhileLoopsToFor(module);
  // Folds the uplift's leftovers (the unused exit computations among them).
  PassManager cleanup(&context);
  cleanup.addPass(createCanonicalizerPass());
  if (failed(cleanup.run(module.getOperation())))
    return failure();

  replacePoisonWithZeros(module);
  return success();
}

LogicalResult Backend::applyPasses(MLIRContext &context, ModuleOp module,
                                   Language language) {
  auto m_result = LogicalResult::success();

  m_ttir.clear();
  m_ttgir.clear();
  m_llir.clear();
  m_llvmir.clear();
  m_asm.clear();

  if (language == Language::TRITON) {
    m_result = makeTTIR(context, module);
    CHECK_RESULT(m_result, "Failed to make TTIR module. Aborting translation.");

    // Triton's canonicalization inside makeTTIR can produce degenerate
    // cf.cond_br %c, ^bb, ^bb operations (both arms the same block).
    // Fold these to cf.br ^bb before makeTTGIR sees them.
    foldDegenerateCondBranches(module);
    // Clean up any dead values created by the fold (e.g., unused comparison
    // results after a degenerate cond_br is replaced with an unconditional br).
    // NOTE: SimplifyPassThroughCondBranch in the canonicalizer can collapse
    // pass-through trampolines into cond_br successors, which may recreate
    // degenerate cond_brs.  Run the fold a second time to catch those.
    {
      PassManager cleanup_pm(&context);
      cleanup_pm.addPass(createCanonicalizerPass());
      auto r = cleanup_pm.run(module.getOperation());
      (void)r; // best-effort
    }
    foldDegenerateCondBranches(module);

    llvm::raw_string_ostream ttir_os(m_ttir);
    module.print(ttir_os);
    ttir_os.flush();

    m_result = makeTTGIR(context, module);
    CHECK_RESULT(m_result,
                 "Failed to make TTGIR module. Aborting translation.");

    llvm::raw_string_ostream ttgir_os(m_ttgir);
    module.print(ttgir_os);
  } else {
    m_result = gluonToTTGIR(context, module);
    CHECK_RESULT(m_result, "Failed to convert GLUON module to TTGIR module. "
                           "Aborting translation.");

    llvm::raw_string_ostream ttgir_os(m_ttgir);
    module.print(ttgir_os);
  }

  m_result = makeLLIR(context, module);
  CHECK_RESULT(m_result, "Failed to make LLIR module. Aborting translation.");

  llvm::raw_string_ostream llir_os(m_llir);
  module.print(llir_os);

  m_result = makeLLVMIR(context, module);
  CHECK_RESULT(m_result, "Failed to make LLVMIR module. Aborting translation.");

  return LogicalResult::success();
}

void Backend::printIR(std::string stage, ModuleOp module) {
  llvm::outs() << "--------------------------------\n";
  llvm::outs() << "Stage: " << stage << "\n";
  module.print(llvm::outs());
  llvm::outs() << "\n--------------------------------\n";
}

std::optional<Error> Backend::addPass(PassManager &pm, MlirPass pass) {
  auto pass_fn = m_pass_fns.find(pass);
  if (pass_fn == m_pass_fns.end()) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid triton pass";
    return m_last_error;
  }

  pm.addPass(pass_fn->second());
  return std::nullopt;
}

std::optional<Error> Backend::addPass(PassManager &pm, MlirPass pass,
                                      int arg0) {
  m_last_error = std::nullopt;
  m_last_error_string = "";

  switch (pass) {
  case MlirPass::ttgpuir_assign_latencies:
    pm.addPass(createTritonGPUAssignLatencies({arg0}));
    break;

  case MlirPass::ttgpuir_warp_specialize:
    pm.addPass(createTritonGPUAutomaticWarpSpecialization({arg0}));
    break;

  default:
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid triton pass";
    break;
  }

  return m_last_error;
}

std::optional<Error> Backend::addPass(PassManager &pm, MlirPass pass,
                                      bool arg0) {
  if (pass != MlirPass::ttgpuir_optimize_dot_operands) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid triton pass";
    return m_last_error;
  }

  pm.addPass(createTritonGPUOptimizeDotOperands({arg0}));
  return std::nullopt;
}

std::optional<Error> Backend::addPass(PassManager &pm, MlirPass pass, int arg0,
                                      bool arg1) {
  if (pass != MlirPass::ttgpuir_pipeline) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid triton pass";
    return m_last_error;
  }

  pm.addPass(createTritonGPUPipeline({arg0, arg1}));
  return std::nullopt;
}

std::optional<Error> Backend::addPass(PassManager &pm, MlirPass pass,
                                      const std::string &arg0, int arg1,
                                      int arg2, int arg3) {
  if (pass != MlirPass::ttir_convert_to_ttgpuir) {
    m_last_error = std::make_optional(Error::InvalidPass);
    m_last_error_string = "Invalid triton pass";
    return m_last_error;
  }

  pm.addPass(createConvertTritonToTritonGPU({arg0, arg1, arg2, arg3}));
  return std::nullopt;
}
} // namespace triton
} // namespace mlir
