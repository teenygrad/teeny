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

#ifndef TRITON_RISCV_BACKEND_H
#define TRITON_RISCV_BACKEND_H

#include <memory>
#include <set>
#include <stdint.h>
#include <string>

#include "llvm/Target/TargetMachine.h"
#include "llvm/TargetParser/Triple.h"

#include "CpuBackend.h"

namespace mlir {
namespace triton {

// ---------------------------------------------------------------------------
// RISC-V backend compile options (FFI-safe / repr(C))
//
// Mirrors the FFI-safe struct conventions used by CudaCompileOptions in
// CudaBackend.h.
// ---------------------------------------------------------------------------

/// FFI-safe compilation options for the RISC-V backend.
struct RiscvCompileOptions {
  const char *target_triple; /// RISC-V target triple
  const char *cpu;           /// RISC-V CPU
  const char *features;      /// RISC-V features
  bool debug;
};

/// RISC-V backend. CpuBackend lowers the incoming Triton module through
/// TritonCPU IR to optimised LLVM IR; this class describes the RISC-V target
/// and compiles that IR through LLVM's RISC-V backend -- makeBIN links the
/// result into a shared library via `ld.lld` so it can be dlopen'd and run.
class RiscvBackend : public CpuBackend {
public:
  RiscvBackend(std::string target, RiscvCompileOptions options);

  virtual ~RiscvBackend();

  virtual void loadDialects(MLIRContext &context) override;

  virtual LogicalResult makeASM(MLIRContext &context, ModuleOp module) override;

  virtual LogicalResult makeBIN(MLIRContext &context, ModuleOp module) override;

protected:
  virtual std::string getTargetArch() const override;

  virtual std::set<std::string> getTargetFeatures() const override;

  /// Uses a real, generic LLVM cpu name matching the triple's width rather
  /// than `m_cpu` -- see the comment in the .cpp for why forwarding that
  /// Triton-side chip identifier directly would abort the process instead of
  /// failing gracefully.
  virtual std::unique_ptr<llvm::TargetMachine> createTargetMachine() override;

private:
  /// The normalized target triple, defaulting to riscv64 when none was given.
  llvm::Triple targetTriple() const;

  /// The LLVM feature string the target machine is created with, e.g.
  /// "+m,+a,+f,+d,+c". getTargetFeatures() derives from the same string, so
  /// pass selection and codegen always agree on the target.
  std::string llvmFeatures() const;

  /// Reparses `m_llvmir` (populated by makeLLVMIR) into `context`, logging
  /// why and returning nullptr on failure.
  std::unique_ptr<llvm::Module> parseStoredLLVMIR(llvm::LLVMContext &context);

  /// Locates the `ld.lld` binary used to link makeBIN's object file into a
  /// shared library: `$TEENYC_LLD_PATH` if set, else the `rust-lld` copy
  /// bundled with this running `teenyc`'s own toolchain (see the .cpp),
  /// else the first `ld.lld` on `PATH` (e.g. a separately apt-installed
  /// `lld` package) as a last resort. Returns an empty string if none are
  /// found.
  std::string findLld();

  // Copied out of the incoming RiscvCompileOptions in the constructor rather
  // than storing that struct (and its raw pointers) by value: the pointers
  // are only guaranteed valid for the duration of the constructor call (see
  // MlirModule::_ffi_strings on the Rust side), so this backend needs its
  // own independent storage for as long as it lives. Empty string means
  // "not set" (mirrors a NULL pointer in RiscvCompileOptions).
  std::string m_target_triple;
  std::string m_cpu;
  std::string m_features;
};

} // namespace triton
} // namespace mlir

#endif /*! TRITON_RISCV_BACKEND_H */
