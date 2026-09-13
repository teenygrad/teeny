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

#ifndef TRITON_CPU_BACKEND_H
#define TRITON_CPU_BACKEND_H

#include <memory>
#include <set>
#include <string>
#include <unordered_map>

#include "llvm/Target/TargetMachine.h"

#include "cpu/include/TritonCPUToLLVM/Passes.h"

#include "Backend.h"

namespace mlir {
namespace triton {

/// Passes only the TritonCPU pipeline uses, ported from triton-cpu's
/// third_party/cpu/backend/compiler.py. Generic MLIR passes the pipeline also
/// needs live in MlirPass (Backend.h) so other backends can share them.
enum CpuPass {
  // ttcir (TritonToTritonCPU)
  ttcpuir_scalarize,
  ttcpuir_convert_memory_ops,
  ttcpuir_convert_ptr_ops,
  ttcpuir_convert_elementwise_ops,
  ttcpuir_convert_elem_manip_ops,
  ttcpuir_convert_dot_op,
  ttcpuir_convert_histogram_op,
  ttcpuir_convert_reduction_op,
  ttcpuir_convert_scan_op,
  ttcpuir_convert_cf_ops,
  ttcpuir_convert_atomic_ops,
  ttcpuir_convert_debug_ops,

  // tttcir (TritonCPUTransforms)
  ttcpuir_canonicalizer,
  ttcpuir_optimize_masks,
  ttcpuir_convert_dot_product,
  ttcpuir_convert_dot_to_amx,
  ttcpuir_convert_dot_to_fma,
  ttcpuir_convert_dot_generic,
  ttcpuir_convert_dot_to_nanokernel,
  ttcpuir_convert_unsupported_ops,
  ttcpuir_decompose_fp_conversions,
  ttcpuir_unroll_and_reorder_elementwise_ops,

  // llir (TritonCPUToLLVM, plus the vector lowerings that take options)
  ttcpuir_lower_vector_multi_dim,
  ttcpuir_vector_to_scf,
  ttcpuir_func_op_to_llvmir,
  ttcpuir_program_id_to_llvmir,
  ttcpuir_memory_op_to_llvmir,
  ttcpuir_atomic_ops_to_llvmir,
  ttcpuir_debug_ops_to_llvmir,
  ttcpuir_math_to_vec_lib,
  ttcpuir_vector_to_llvmir,
};

/// Lowers Triton IR for CPU targets through TritonCPU IR to optimised LLVM
/// IR, following triton-cpu's compiler.py stage by stage. Abstract: a
/// subclass describes its target (arch, features, TargetMachine) and owns
/// codegen and linking (makeASM/makeBIN).
///
/// triton-cpu has two stages Backend has no slot for, ttcir and tttcir. They
/// are makeTTCIR and makeTTTCIR here, run in that order from makeTTGIR, so
/// Backend::applyPasses stays shared with the GPU backends.
class CpuBackend : public Backend {
public:
  CpuBackend(std::string target, bool debug);

  virtual ~CpuBackend();

  virtual void loadDialects(MLIRContext &context) override;

  /// compiler.py make_ttir.
  virtual LogicalResult makeTTIR(MLIRContext &context,
                                 ModuleOp module) override;

  /// makeTTCIR followed by makeTTTCIR, so getTTGIR() holds the
  /// target-specialised TritonCPU IR and getTTCIR() the IR in between.
  virtual LogicalResult makeTTGIR(MLIRContext &context,
                                  ModuleOp module) override;

  /// Gluon has no CPU lowering; always fails with Error::NotImplemented.
  virtual LogicalResult gluonToTTGIR(MLIRContext &context,
                                     ModuleOp module) override;

  /// The pass pipeline of compiler.py make_llir: TritonCPU IR to the MLIR
  /// LLVM dialect.
  virtual LogicalResult makeLLIR(MLIRContext &context,
                                 ModuleOp module) override;

  /// The rest of make_llir: translation to an llvm::Module, target triple and
  /// data layout from createTargetMachine(), then the O3 pipeline.
  virtual LogicalResult makeLLVMIR(MLIRContext &context,
                                   ModuleOp module) override;

  /// compiler.py make_ttcir: Triton IR to TritonCPU IR.
  virtual LogicalResult makeTTCIR(MLIRContext &context, ModuleOp module);

  /// compiler.py make_tttcir: target-dependent TritonCPU IR optimisation,
  /// chiefly how tt.dot is lowered.
  virtual LogicalResult makeTTTCIR(MLIRContext &context, ModuleOp module);

  const char *getTTCIR() const { return m_ttcir.c_str(); }

protected:
  /// LLVM arch name of the target, e.g. "x86_64", "riscv64" or "aarch64".
  virtual std::string getTargetArch() const = 0;

  /// LLVM feature names enabled on the target, without "+"/"-" prefixes.
  /// triton-cpu chooses passes from the host CPU's features; taking them from
  /// the subclass keeps that choice correct when cross-compiling.
  virtual std::set<std::string> getTargetFeatures() const = 0;

  /// TargetMachine for the target, used for the LLVM module's triple, data
  /// layout and O3 pipeline. Returns nullptr, after logging why, on failure.
  virtual std::unique_ptr<llvm::TargetMachine> createTargetMachine() = 0;

  std::optional<Error> addCpuPass(PassManager &pm, CpuPass pass);

  std::optional<Error> addCpuPass(PassManager &pm, CpuPass pass, bool arg0);

  std::optional<Error> addCpuPass(PassManager &pm, CpuPass pass, bool arg0,
                                  bool arg1);

  std::optional<Error> addCpuPass(PassManager &pm, CpuPass pass, bool arg0,
                                  bool arg1, bool arg2);

  std::optional<Error> addCpuPass(PassManager &pm, CpuPass pass, bool arg0,
                                  unsigned arg1, bool arg2);

  /// Takes std::string rather than const char * so a string literal can never
  /// silently bind to one of the bool overloads above.
  std::optional<Error> addCpuPass(PassManager &pm, CpuPass pass,
                                  const std::string &arg0);

  std::optional<Error> addCpuPass(PassManager &pm, CpuPass pass,
                                  cpu::VecLib arg0,
                                  const std::set<std::string> &arg1);

  bool m_debug;

  /// TritonCPU IR after makeTTCIR, before makeTTTCIR specialises it.
  std::string m_ttcir;

private:
  std::optional<Error> invalidCpuPass();

  /// The TritonCPU pass factories return unique_ptr<OperationPass<ModuleOp>>
  /// rather than unique_ptr<Pass>, so each entry adapts through a lambda.
  std::unordered_map<CpuPass, std::unique_ptr<Pass> (*)()> m_cpu_pass_fns;
};

} // namespace triton
} // namespace mlir

#endif /*! TRITON_CPU_BACKEND_H */
