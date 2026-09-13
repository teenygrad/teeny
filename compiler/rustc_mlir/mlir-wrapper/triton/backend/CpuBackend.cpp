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

#include <algorithm>

#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/OptimizationLevel.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/Conversion/Passes.h"
#include "mlir/Conversion/VectorToSCF/VectorToSCF.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

// Same include order as triton-cpu's triton_cpu.cc.
#include "cpu/include/ScalarizePass/ScalarizeInterfaceImpl.h"
#include "cpu/include/TritonCPUToLLVM/Passes.h"
#include "cpu/include/TritonCPUTransforms/Passes.h"
#include "cpu/include/TritonToTritonCPU/Passes.h"

#include "cpu/include/Dialect/TritonCPU/IR/Dialect.h"

#include "CpuBackend.h"

namespace mlir {
namespace triton {

namespace {

// triton-cpu reads these from CPUOptions or TRITON_CPU_* environment
// variables; they are fixed at triton-cpu's defaults here.
constexpr bool kAssumeInBounds = false;         // TRITON_CPU_ASSUME_IN_BOUNDS
constexpr bool kEnableFastMath = true;          // TRITON_CPU_FAST_MATH
constexpr bool kDotProductHorizontalSum = true; // TRITON_CPU_DOT_PROD_HORIZ_SUM
constexpr bool kUnrollAndReorderElementwiseOps =
    false;                            // TRITON_CPU_UNROLL_AND_REORDER_...
constexpr bool kEmitLineInfo = true;  // !TRITON_DISABLE_LINE_INFO
constexpr cpu::VecLib kVecLib = cpu::VecLib::Sleef; // CPUOptions.vec_lib

/// triton-cpu only maps math onto a vector library the target can actually
/// call into: libsleef needs NEON, SSE or AVX, libmvec needs AVX-512.
bool vecLibSupported(cpu::VecLib lib, const std::set<std::string> &features) {
  static const std::set<std::string> sleefFeatures = {"neon", "sse", "avx"};
  static const std::set<std::string> mvecFeatures = {"avx512f"};
  const auto &required =
      lib == cpu::VecLib::Sleef ? sleefFeatures : mvecFeatures;
  return std::any_of(
      required.begin(), required.end(),
      [&](const std::string &feature) { return features.count(feature) != 0; });
}

std::string joinFeatures(const std::set<std::string> &features) {
  std::string joined;
  for (const auto &feature : features) {
    if (!joined.empty()) {
      joined += ",";
    }
    joined += feature;
  }
  return joined;
}

/// Equivalent of the llvm.optimize_module(mod, O3) triton-cpu runs. CudaBackend
/// deliberately skips host O3 because it has no NVPTX TargetMachine; a CPU
/// target does have one, and the pipeline needs it for correct vectorization
/// and cost decisions.
void optimizeModule(llvm::Module &module, llvm::TargetMachine &tm) {
  llvm::LoopAnalysisManager lam;
  llvm::FunctionAnalysisManager fam;
  llvm::CGSCCAnalysisManager cgam;
  llvm::ModuleAnalysisManager mam;

  llvm::PipelineTuningOptions tuningOptions;
  tuningOptions.LoopUnrolling = true;
  tuningOptions.LoopInterleaving = true;
  tuningOptions.LoopVectorization = true;
  tuningOptions.SLPVectorization = true;

  llvm::PassBuilder pb(&tm, tuningOptions);
  pb.registerModuleAnalyses(mam);
  pb.registerCGSCCAnalyses(cgam);
  pb.registerFunctionAnalyses(fam);
  pb.registerLoopAnalyses(lam);
  pb.crossRegisterProxies(lam, fam, cgam, mam);

  llvm::ModulePassManager mpm =
      pb.buildPerModuleDefaultPipeline(llvm::OptimizationLevel::O3);
  mpm.run(module, mam);
}

} // namespace

CpuBackend::CpuBackend(std::string target, bool debug)
    : Backend(target), m_debug(debug),
      m_cpu_pass_fns({
          // ttcir
          {ttcpuir_convert_ptr_ops,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createConvertPtrOps();
           }},
          {ttcpuir_convert_elementwise_ops,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createConvertElementwiseOps();
           }},
          {ttcpuir_convert_elem_manip_ops,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createConvertElemManipOps();
           }},
          {ttcpuir_convert_dot_op,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createConvertDotOp();
           }},
          {ttcpuir_convert_histogram_op,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createConvertHistogramOp();
           }},
          {ttcpuir_convert_scan_op,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createConvertScanOp();
           }},
          {ttcpuir_convert_cf_ops,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createConvertControlFlowOps();
           }},
          {ttcpuir_convert_atomic_ops,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createConvertAtomicOps();
           }},
          {ttcpuir_convert_debug_ops,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createConvertDebugOps();
           }},

          // tttcir
          {ttcpuir_canonicalizer,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createCanonicalize();
           }},
          {ttcpuir_optimize_masks,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createOptimizeMasks();
           }},
          {ttcpuir_convert_dot_to_fma,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createConvertDotToFMA();
           }},
          {ttcpuir_convert_dot_generic,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createConvertDotGeneric();
           }},

          // llir
          {ttcpuir_func_op_to_llvmir,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createFuncOpToLLVMPass();
           }},
          {ttcpuir_program_id_to_llvmir,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createGetProgramIdOpToLLVMPass();
           }},
          {ttcpuir_memory_op_to_llvmir,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createMemoryOpToLLVMPass();
           }},
          {ttcpuir_atomic_ops_to_llvmir,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createAtomicOpsToLLVMPass();
           }},
          {ttcpuir_debug_ops_to_llvmir,
           +[]() -> std::unique_ptr<Pass> {
             return cpu::createDebugOpsToLLVMPass();
           }},
      }) {}

CpuBackend::~CpuBackend() {}

void CpuBackend::loadDialects(MLIRContext &context) {
  // Mirrors triton-cpu's load_dialects. The LLVM IR translations makeLLVMIR
  // depends on are already registered on this context by the Rust side
  // (register_all_llvm_translations), so they are not registered again here.
  DialectRegistry registry;
  registry.insert<cpu::TritonCPUDialect, vector::VectorDialect>();
  cpu::registerTritonOpScalarizeExternalModels(registry);
  context.appendDialectRegistry(registry);
  context.loadAllAvailableDialects();
}

LogicalResult CpuBackend::makeTTIR(MLIRContext &context, ModuleOp module) {
  PassManager pm(&context);
  if (m_debug) {
    pm.enableIRPrinting();
  }

  addPass(pm, MlirPass::inliner);
  addPass(pm, MlirPass::ttir_combine);
  addPass(pm, MlirPass::canonicalizer);
  addPass(pm, MlirPass::ttir_reorder_broadcast);
  addPass(pm, MlirPass::cse);
  addPass(pm, MlirPass::licm);
  addPass(pm, MlirPass::symbol_dce);

  return pm.run(module.getOperation());
}

LogicalResult CpuBackend::makeTTGIR(MLIRContext &context, ModuleOp module) {
  auto result = makeTTCIR(context, module);
  CHECK_RESULT(result, "Failed to make TTCIR module. Aborting translation.");

  m_ttcir.clear();
  llvm::raw_string_ostream ttcir_os(m_ttcir);
  module.print(ttcir_os);
  ttcir_os.flush();

  result = makeTTTCIR(context, module);
  CHECK_RESULT(result, "Failed to make TTTCIR module. Aborting translation.");

  return success();
}

LogicalResult CpuBackend::gluonToTTGIR(MLIRContext &context, ModuleOp module) {
  m_last_error = std::make_optional(Error::NotImplemented);
  m_last_error_string = "Gluon is not supported by the CPU backend";
  llvm::errs() << "CpuBackend: " << m_last_error_string << "\n";
  return failure();
}

LogicalResult CpuBackend::makeTTCIR(MLIRContext &context, ModuleOp module) {
  PassManager pm(&context);
  if (m_debug) {
    pm.enableIRPrinting();
  }

  addCpuPass(pm, CpuPass::ttcpuir_scalarize, /*skipGatherScatter=*/true);
  addCpuPass(pm, CpuPass::ttcpuir_convert_memory_ops,
             /*useGatherScatter=*/true, kAssumeInBounds);
  addCpuPass(pm, CpuPass::ttcpuir_convert_ptr_ops);
  addCpuPass(pm, CpuPass::ttcpuir_convert_elementwise_ops);
  addCpuPass(pm, CpuPass::ttcpuir_convert_elem_manip_ops);
  addCpuPass(pm, CpuPass::ttcpuir_convert_dot_op);
  addCpuPass(pm, CpuPass::ttcpuir_convert_histogram_op);
  addCpuPass(pm, CpuPass::ttcpuir_convert_reduction_op,
             /*useReductionOp=*/true, /*useMultiDimReductionOp=*/false);
  addCpuPass(pm, CpuPass::ttcpuir_convert_scan_op);
  addCpuPass(pm, CpuPass::ttcpuir_convert_cf_ops);
  addCpuPass(pm, CpuPass::ttcpuir_convert_atomic_ops);
  addCpuPass(pm, CpuPass::ttcpuir_convert_debug_ops);
  addPass(pm, MlirPass::cse);
  addPass(pm, MlirPass::symbol_dce);
  addPass(pm, MlirPass::canonicalizer);

  return pm.run(module.getOperation());
}

LogicalResult CpuBackend::makeTTTCIR(MLIRContext &context, ModuleOp module) {
  const std::string arch = getTargetArch();
  const std::set<std::string> features = getTargetFeatures();
  auto has = [&](const char *feature) { return features.count(feature) != 0; };
  const std::string featureList = joinFeatures(features);

  PassManager pm(&context);
  if (m_debug) {
    pm.enableIRPrinting();
  }

  addCpuPass(pm, CpuPass::ttcpuir_canonicalizer);
  addCpuPass(pm, CpuPass::ttcpuir_optimize_masks);
  addPass(pm, MlirPass::canonicalizer);

  // triton-cpu takes the ukernel path when oneDNN or libxsmm is available.
  // Both are compiled out of this build, so the nanokernel lowering is always
  // used; it only rewrites when the features include the AVX/AMX extensions
  // it targets.
  addCpuPass(pm, CpuPass::ttcpuir_convert_dot_to_nanokernel, featureList);

  if ((arch == "aarch64" || arch == "armv8") && has("fp-armv8") &&
      has("neon")) {
    addCpuPass(pm, CpuPass::ttcpuir_convert_dot_product,
               kDotProductHorizontalSum);
  }
  if (has("amx-tile")) {
    // FP16 has no lowering in the x86 dialect's AMX ops yet, as in triton-cpu.
    addCpuPass(pm, CpuPass::ttcpuir_convert_dot_to_amx, has("amx-int8"),
               /*convertFp16=*/false, has("amx-bf16"));
  }
  if (has("avx512f")) {
    addCpuPass(pm, CpuPass::ttcpuir_convert_dot_to_fma);
  }
  addCpuPass(pm, CpuPass::ttcpuir_convert_dot_generic);

  // Only x86 can lack native BF16, hence the arch check; there is no lowering
  // for mixed-precision matmul or for FP8/FP16/BF16 math library calls, so
  // those are always converted.
  const bool lacksNativeBf16 = arch == "x86_64" && !has("avx512bf16");
  addCpuPass(pm, CpuPass::ttcpuir_convert_unsupported_ops,
             /*promoteBf16ToFp32=*/lacksNativeBf16,
             /*convertMixedPrecisionMatmul=*/true,
             /*promoteLibMathToFp32=*/true);
  addCpuPass(pm, CpuPass::ttcpuir_decompose_fp_conversions,
             /*decomposeBf16Conversions=*/lacksNativeBf16,
             /*decomposeFp8Conversions=*/true);
  if (kUnrollAndReorderElementwiseOps) {
    addCpuPass(pm, CpuPass::ttcpuir_unroll_and_reorder_elementwise_ops,
               featureList);
  }
  addPass(pm, MlirPass::cse);
  addPass(pm, MlirPass::symbol_dce);
  addPass(pm, MlirPass::canonicalizer);

  return pm.run(module.getOperation());
}

LogicalResult CpuBackend::makeLLIR(MLIRContext &context, ModuleOp module) {
  const std::string arch = getTargetArch();
  const std::set<std::string> features = getTargetFeatures();

  PassManager pm(&context);
  if (m_debug) {
    pm.enableIRPrinting();
  }

  addCpuPass(pm, CpuPass::ttcpuir_lower_vector_multi_dim);
  addPass(pm, MlirPass::memref_expand_strided_metadata);
  addCpuPass(pm, CpuPass::ttcpuir_vector_to_scf, /*fullUnroll=*/true,
             /*targetRank=*/1u, /*lowerTensors=*/false);
  addPass(pm, MlirPass::lower_affine);
  addPass(pm, MlirPass::scf_to_cf);
  addPass(pm, MlirPass::index_to_llvmir);
  addCpuPass(pm, CpuPass::ttcpuir_func_op_to_llvmir);
  addCpuPass(pm, CpuPass::ttcpuir_program_id_to_llvmir);
  addCpuPass(pm, CpuPass::ttcpuir_memory_op_to_llvmir);
  addCpuPass(pm, CpuPass::ttcpuir_atomic_ops_to_llvmir);
  addCpuPass(pm, CpuPass::ttcpuir_debug_ops_to_llvmir);

  if (vecLibSupported(kVecLib, features)) {
    addCpuPass(pm, CpuPass::ttcpuir_math_to_vec_lib, kVecLib, features);
  }

  addPass(pm, MlirPass::math_to_llvmir);
  addPass(pm, MlirPass::math_to_libm);
  // The x86 vector-to-LLVM lowering emits x86-specific intrinsics, so it can
  // only be enabled for x86 targets.
  addCpuPass(pm, CpuPass::ttcpuir_vector_to_llvmir, kEnableFastMath,
             /*x86=*/arch == "x86_64");
  addPass(pm, MlirPass::memref_to_llvmir);
  addPass(pm, MlirPass::reconcile_unrealized_casts);
  addPass(pm, MlirPass::arith_to_llvmir);
  addPass(pm, MlirPass::func_to_llvmir);
  addPass(pm, MlirPass::ub_to_llvmir);
  addPass(pm, MlirPass::canonicalizer);
  addPass(pm, MlirPass::cse);
  addPass(pm, MlirPass::symbol_dce);
  if (kEmitLineInfo) {
    addPass(pm, MlirPass::llvmir_di_scope);
  }

  return pm.run(module.getOperation());
}

LogicalResult CpuBackend::makeLLVMIR(MLIRContext &context, ModuleOp module) {
  auto tm = createTargetMachine();
  if (!tm) {
    return failure();
  }

  llvm::LLVMContext llvmContext;
  auto llvmMod =
      mlir::translateModuleToLLVMIR(module.getOperation(), llvmContext);
  if (!llvmMod) {
    llvm::errs() << "CpuBackend: failed to translate MLIR module to LLVM IR\n";
    return failure();
  }

  llvmMod->setTargetTriple(tm->getTargetTriple());
  llvmMod->setDataLayout(tm->createDataLayout());

  optimizeModule(*llvmMod, *tm);

  m_llvmir.clear();
  llvm::raw_string_ostream os(m_llvmir);
  llvmMod->print(os, nullptr);

  return success();
}

std::optional<Error> CpuBackend::invalidCpuPass() {
  m_last_error = std::make_optional(Error::InvalidPass);
  m_last_error_string = "Invalid TritonCPU pass";
  return m_last_error;
}

std::optional<Error> CpuBackend::addCpuPass(PassManager &pm, CpuPass pass) {
  m_last_error = std::nullopt;
  m_last_error_string = "";

  if (pass == CpuPass::ttcpuir_lower_vector_multi_dim) {
    // Anchored on tt.func rather than the module, as in triton-cpu.
    pm.addNestedPass<triton::FuncOp>(cpu::createLowerMultiReductionPass());
    return std::nullopt;
  }

  auto pass_fn = m_cpu_pass_fns.find(pass);
  if (pass_fn == m_cpu_pass_fns.end()) {
    return invalidCpuPass();
  }

  pm.addPass(pass_fn->second());
  return std::nullopt;
}

std::optional<Error> CpuBackend::addCpuPass(PassManager &pm, CpuPass pass,
                                            bool arg0) {
  m_last_error = std::nullopt;
  m_last_error_string = "";

  switch (pass) {
  case CpuPass::ttcpuir_scalarize:
    pm.addPass(cpu::createScalarizeUsingForOpPass(arg0));
    break;

  case CpuPass::ttcpuir_convert_dot_product:
    pm.addPass(cpu::createConvertDotProduct(arg0));
    break;

  default:
    return invalidCpuPass();
  }

  return std::nullopt;
}

std::optional<Error> CpuBackend::addCpuPass(PassManager &pm, CpuPass pass,
                                            bool arg0, bool arg1) {
  m_last_error = std::nullopt;
  m_last_error_string = "";

  switch (pass) {
  case CpuPass::ttcpuir_convert_memory_ops:
    pm.addPass(cpu::createConvertMemoryOps(arg0, arg1));
    break;

  case CpuPass::ttcpuir_convert_reduction_op:
    pm.addPass(cpu::createConvertReductionOp(arg0, arg1));
    break;

  case CpuPass::ttcpuir_decompose_fp_conversions:
    pm.addPass(cpu::createDecomposeFpConversions(arg0, arg1));
    break;

  case CpuPass::ttcpuir_vector_to_llvmir: {
    ConvertVectorToLLVMPassOptions opts;
    opts.reassociateFPReductions = arg0;
    opts.x86 = arg1;
    // As in triton-cpu: the Dot lowering is very slow to compile and the
    // Matmul lowering fails to split llvm.matrix.multiply, which leaves
    // OuterProduct as the dependable CPU lowering.
    opts.vectorContractLowering = vector::VectorContractLowering::OuterProduct;
    pm.addPass(createConvertVectorToLLVMPass(opts));
    break;
  }

  default:
    return invalidCpuPass();
  }

  return std::nullopt;
}

std::optional<Error> CpuBackend::addCpuPass(PassManager &pm, CpuPass pass,
                                            bool arg0, bool arg1, bool arg2) {
  m_last_error = std::nullopt;
  m_last_error_string = "";

  switch (pass) {
  case CpuPass::ttcpuir_convert_dot_to_amx:
    pm.addPass(cpu::createConvertDotToAMX(arg0, arg1, arg2));
    break;

  case CpuPass::ttcpuir_convert_unsupported_ops:
    pm.addPass(cpu::createConvertUnsupportedOps(arg0, arg1, arg2));
    break;

  default:
    return invalidCpuPass();
  }

  return std::nullopt;
}

std::optional<Error> CpuBackend::addCpuPass(PassManager &pm, CpuPass pass,
                                            bool arg0, unsigned arg1,
                                            bool arg2) {
  m_last_error = std::nullopt;
  m_last_error_string = "";

  if (pass != CpuPass::ttcpuir_vector_to_scf) {
    return invalidCpuPass();
  }

  VectorTransferToSCFOptions opts;
  opts.enableFullUnroll(arg0);
  opts.setTargetRank(arg1);
  opts.enableLowerTensors(arg2);
  pm.addPass(createConvertVectorToSCFPass(opts));
  return std::nullopt;
}

std::optional<Error> CpuBackend::addCpuPass(PassManager &pm, CpuPass pass,
                                            const std::string &arg0) {
  m_last_error = std::nullopt;
  m_last_error_string = "";

  switch (pass) {
  case CpuPass::ttcpuir_convert_dot_to_nanokernel:
    pm.addPass(cpu::createConvertDotToNanokernel(arg0));
    break;

  case CpuPass::ttcpuir_unroll_and_reorder_elementwise_ops:
    pm.addPass(cpu::createUnrollAndReorderElementwiseOps(arg0));
    break;

  default:
    return invalidCpuPass();
  }

  return std::nullopt;
}

std::optional<Error>
CpuBackend::addCpuPass(PassManager &pm, CpuPass pass, cpu::VecLib arg0,
                       const std::set<std::string> &arg1) {
  m_last_error = std::nullopt;
  m_last_error_string = "";

  if (pass != CpuPass::ttcpuir_math_to_vec_lib) {
    return invalidCpuPass();
  }

  pm.addPass(cpu::createMathToVecLibPass(arg0, arg1));
  return std::nullopt;
}

} // namespace triton
} // namespace mlir
