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

#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Verifier.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Triple.h"

#include "riscv/Dialect/RVV/IR/Dialect.h"

#include "RiscvBackend.h"

#include <cstdlib>

namespace mlir {
namespace triton {

namespace {

/// Name for the placeholder kernel function makeLLVMIR synthesizes: the
/// incoming module's symbol name if it has one, else a generic fallback.
std::string kernelNameFor(ModuleOp module) {
  if (auto name = module.getName()) {
    return name->str();
  }
  return "riscv_kernel";
}

} // namespace

RiscvBackend::RiscvBackend(std::string target, RiscvCompileOptions options)
    : Backend(target),
      m_target_triple(options.target_triple ? options.target_triple : ""),
      m_cpu(options.cpu ? options.cpu : ""),
      m_features(options.features ? options.features : ""),
      m_debug(options.debug) {}

RiscvBackend::~RiscvBackend() {}

void RiscvBackend::loadDialects(MLIRContext &context) {
  // Registered for future use: this backend does not yet lower the
  // incoming Triton/MLIR module through any dialect at all (see
  // makeTTIR/makeTTGIR/makeLLIR below) -- makeLLVMIR synthesizes a
  // placeholder kernel function directly. RVVDialect exists so real
  // MIR-to-LLVM-IR lowering has somewhere to represent RVV-specific
  // concepts (see RVVDialect.td) once that lowering exists.
  DialectRegistry registry;
  registry.insert<mlir::rvv::RVVDialect>();
  context.appendDialectRegistry(registry);
}

LogicalResult RiscvBackend::makeTTIR(MLIRContext &context, ModuleOp module) {
  // This backend doesn't lower the incoming module yet -- makeLLVMIR
  // synthesizes a placeholder kernel function directly instead (see its
  // comment) -- so there is no TTIR-stage work to do until it does.
  return success();
}

LogicalResult RiscvBackend::makeTTGIR(MLIRContext &context, ModuleOp module) {
  // See makeTTIR: no TTGIR-stage work until this backend lowers the real
  // module instead of synthesizing a placeholder in makeLLVMIR.
  return success();
}

LogicalResult RiscvBackend::gluonToTTGIR(MLIRContext &context,
                                         ModuleOp module) {
  // NOP for RISC-V backend
  return success();
}

LogicalResult RiscvBackend::makeLLIR(MLIRContext &context, ModuleOp module) {
  // See makeTTIR: no LLIR-stage work until this backend lowers the real
  // module instead of synthesizing a placeholder in makeLLVMIR.
  return success();
}

LogicalResult RiscvBackend::makeLLVMIR(MLIRContext &context, ModuleOp module) {
  // Real MIR/Triton-to-LLVM-IR lowering for RISC-V doesn't exist yet (see
  // makeTTIR/makeTTGIR/makeLLIR above). Until it does, synthesize a minimal
  // placeholder kernel -- a single exported `void @<name>()` function with
  // an empty body -- so makeASM/makeBIN have a real LLVM module to compile
  // through LLVM's RISC-V backend instead of failing outright.
  llvm::LLVMContext llvmContext;
  std::string kernelName = kernelNameFor(module);
  auto llvmMod = std::make_unique<llvm::Module>(kernelName, llvmContext);

  llvm::Triple triple(llvm::Triple::normalize(
      m_target_triple.empty() ? "riscv64" : m_target_triple));
  llvmMod->setTargetTriple(triple);

  auto *funcTy =
      llvm::FunctionType::get(llvm::Type::getVoidTy(llvmContext),
                              /*isVarArg=*/false);
  auto *func = llvm::Function::Create(
      funcTy, llvm::Function::ExternalLinkage, kernelName, llvmMod.get());
  auto *entry = llvm::BasicBlock::Create(llvmContext, "entry", func);
  llvm::IRBuilder<> builder(entry);
  builder.CreateRetVoid();

  std::string verifyError;
  llvm::raw_string_ostream verifyOs(verifyError);
  if (llvm::verifyModule(*llvmMod, &verifyOs)) {
    llvm::errs() << "RiscvBackend: generated placeholder module failed "
                    "verification: "
                 << verifyError << "\n";
    return failure();
  }

  llvm::raw_string_ostream os(m_llvmir);
  llvmMod->print(os, nullptr);

  return success();
}

LogicalResult RiscvBackend::makeASM(MLIRContext &context, ModuleOp module) {
  llvm::TargetMachine *tm = createRiscvTargetMachine();
  if (!tm) {
    return failure();
  }

  llvm::LLVMContext llvmContext;
  auto mod = parseStoredLLVMIR(llvmContext);
  if (!mod) {
    delete tm;
    return failure();
  }

  llvm::SmallVector<char, 0> asmBuf;
  {
    llvm::raw_svector_ostream os(asmBuf);
    llvm::legacy::PassManager pm;
    if (tm->addPassesToEmitFile(pm, os, nullptr,
                                llvm::CodeGenFileType::AssemblyFile)) {
      llvm::errs() << "RiscvBackend: failed to add passes to emit assembly\n";
      delete tm;
      return failure();
    }
    pm.run(*mod);
  }
  delete tm;

  m_asm.assign(asmBuf.data(), asmBuf.size());
  return success();
}

LogicalResult RiscvBackend::makeBIN(MLIRContext &context, ModuleOp module) {
  llvm::TargetMachine *tm = createRiscvTargetMachine();
  if (!tm) {
    return failure();
  }

  llvm::LLVMContext llvmContext;
  auto mod = parseStoredLLVMIR(llvmContext);
  if (!mod) {
    delete tm;
    return failure();
  }

  llvm::SmallVector<char, 0> objBuf;
  {
    llvm::raw_svector_ostream os(objBuf);
    llvm::legacy::PassManager pm;
    if (tm->addPassesToEmitFile(pm, os, nullptr,
                                llvm::CodeGenFileType::ObjectFile)) {
      llvm::errs() << "RiscvBackend: failed to add passes to emit object file\n";
      delete tm;
      return failure();
    }
    pm.run(*mod);
  }
  delete tm;

  // Link the object into a shared library so it can be dlopen'd and run at
  // runtime. Uses LLD as a cross-linker (it can target RISC-V regardless of
  // the host architecture, unlike the host's own `cc`/`ld`).
  llvm::SmallString<128> objPath;
  if (auto ec =
          llvm::sys::fs::createTemporaryFile("riscv_kernel", "o", objPath)) {
    llvm::errs() << "RiscvBackend: failed to create temp object file: "
                 << ec.message() << "\n";
    return failure();
  }
  llvm::SmallString<128> soPath;
  if (auto ec =
          llvm::sys::fs::createTemporaryFile("riscv_kernel", "so", soPath)) {
    llvm::errs() << "RiscvBackend: failed to create temp shared object file: "
                 << ec.message() << "\n";
    llvm::sys::fs::remove(objPath);
    return failure();
  }
  auto removeTemps = llvm::make_scope_exit([&] {
    llvm::sys::fs::remove(objPath);
    llvm::sys::fs::remove(soPath);
  });

  {
    std::error_code ec;
    llvm::raw_fd_ostream objFile(objPath, ec, llvm::sys::fs::OF_None);
    if (ec) {
      llvm::errs() << "RiscvBackend: failed to write temp object file: "
                   << ec.message() << "\n";
      return failure();
    }
    objFile << llvm::StringRef(objBuf.data(), objBuf.size());
  }

  std::string lldPath = findLld();
  if (lldPath.empty()) {
    llvm::errs()
        << "RiscvBackend: could not find `ld.lld` to link the RISC-V shared "
           "library (set $TEENYC_LLD_PATH, or put ld.lld on PATH)\n";
    return failure();
  }
  llvm::errs() << "RiscvBackend: linking with " << lldPath << "\n";

  llvm::SmallVector<llvm::StringRef, 8> args = {
      lldPath, "-shared", "-m", "elf64lriscv", "-o", soPath, objPath};
  std::string errMsg;
  int rc = llvm::sys::ExecuteAndWait(lldPath, args, std::nullopt, {}, 0, 0,
                                     &errMsg);
  if (rc != 0) {
    llvm::errs() << "RiscvBackend: ld.lld failed (exit " << rc
                 << "): " << errMsg << "\n";
    return failure();
  }

  auto soBuf = llvm::MemoryBuffer::getFile(soPath);
  if (!soBuf) {
    llvm::errs() << "RiscvBackend: failed to read linked shared library: "
                 << soBuf.getError().message() << "\n";
    return failure();
  }

  m_bin.assign((*soBuf)->getBufferStart(), (*soBuf)->getBufferEnd());
  return success();
}

llvm::TargetMachine *RiscvBackend::createRiscvTargetMachine() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  llvm::Triple triple(llvm::Triple::normalize(
      m_target_triple.empty() ? "riscv64" : m_target_triple));
  std::string targetError;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple, targetError);
  if (!target) {
    llvm::errs() << "RiscvBackend: " << targetError << "\n";
    return nullptr;
  }

  // `m_cpu` (e.g. `spacemit-k3`, `generic-rvv1.0`) is a
  // Triton/RiscvBackend-side chip identifier, not an LLVM `-mcpu` name --
  // see RiscvCompileOptions in RiscvBackend.h -- and there is no mapping
  // from that vocabulary to a real LLVM cpu/feature string yet. Passing an
  // unrecognized name straight through is not just silently wrong: LLVM's
  // RISC-V backend calls report_fatal_error (aborting the whole process,
  // not a recoverable LogicalResult::failure()) when it can't derive a
  // valid XLen from the cpu, e.g. "LLVM ERROR: RV64 target requires an
  // RV64 CPU". So for now this always uses a real, generic LLVM cpu name
  // matching the triple's width, and ignores m_cpu -- fine for the
  // placeholder `ret void` body makeLLVMIR produces today, but a real
  // chip-name-to-LLVM-cpu/feature mapping is needed before m_cpu
  // can be honored.
  std::string cpu = triple.isArch64Bit() ? "generic-rv64" : "generic-rv32";

  // "generic-rv64"/"generic-rv32" alone imply no ISA extensions, which
  // defaults codegen to the soft-float ABI (lp64/ilp32) -- incompatible
  // with the hard-float ABI (lp64d/ilp32d) essentially all real RISC-V
  // Linux userspace (glibc, other .so's on the system) actually uses.
  // Matches the `features` rustc_target::spec::targets::riscv64_generic
  // declares (M/A/F/D/C, i.e. the standard "G" extension set, plus V);
  // once m_features carries a real per-chip feature string this
  // should prefer that instead of always using this fixed baseline.
  std::string features = "+m,+a,+f,+d,+c";

  // PIC: makeBIN links the resulting object into a shared library.
  llvm::TargetOptions opts;
  llvm::TargetMachine *tm = target->createTargetMachine(
      triple, cpu, features, opts, llvm::Reloc::PIC_, std::nullopt,
      llvm::CodeGenOptLevel::Default);
  if (!tm) {
    llvm::errs() << "RiscvBackend: failed to create target machine for "
                 << triple.getTriple() << " (cpu=" << cpu << ")\n";
  }
  return tm;
}

std::unique_ptr<llvm::Module>
RiscvBackend::parseStoredLLVMIR(llvm::LLVMContext &context) {
  auto buf = llvm::MemoryBuffer::getMemBuffer(m_llvmir, "<riscv-llvm-ir>");
  llvm::SMDiagnostic err;
  auto mod = llvm::parseIR(buf->getMemBufferRef(), err, context);
  if (!mod) {
    err.print("RiscvBackend", llvm::errs());
  }
  return mod;
}

/// Locates the `rust-lld` copy bundled with the running `teenyc`'s own
/// toolchain, at `<prefix>/lib/rustlib/<host-target>/bin/gcc-ld/ld.lld`
/// (present on any standard rustup/cargo-teeny install, precisely so a
/// compiler like this one can self-contained-link without depending on a
/// system linker package). `<prefix>` is derived from the running
/// executable's own path (`<prefix>/bin/teenyc`) rather than assumed, so
/// this works regardless of install location. `<host-target>` is found by
/// scanning `lib/rustlib/*` rather than constructed from a triple string,
/// since only the host's own subdirectory ships `bin/gcc-ld/ld.lld` and its
/// exact spelling doesn't need to match LLVM's triple formatting this way.
///
/// Named `ld.lld` (not the bare `rust-lld` binary one level up) because
/// LLD's driver selects its ELF/Darwin/etc. "flavor" from argv[0]'s
/// basename by default; invoking the bare `rust-lld` binary directly with
/// this backend's plain `-shared -m ... -o ...` arguments (no `-flavor`
/// flag) would not reliably select the ELF driver the way invoking a
/// binary actually named `ld.lld` does.
static std::string findToolchainLld() {
  // Passing this function's own address is the standard LLVM idiom for
  // getMainExecutable's dladdr-based fallback path (used on platforms
  // without a reliable /proc/self/exe equivalent); it never needs to be
  // called.
  std::string exePath = llvm::sys::fs::getMainExecutable(
      nullptr, reinterpret_cast<void *>(&findToolchainLld));
  if (exePath.empty()) {
    return {};
  }

  // exePath is <prefix>/bin/teenyc (or /rustc, /cargo-teeny's teenyc, etc.)
  // -- strip twice to get <prefix>.
  llvm::SmallString<256> rustlibDir(exePath);
  llvm::sys::path::remove_filename(rustlibDir); // drop the executable name
  llvm::sys::path::remove_filename(rustlibDir); // drop "bin"
  llvm::sys::path::append(rustlibDir, "lib", "rustlib");

  std::error_code ec;
  llvm::sys::fs::directory_iterator it(rustlibDir, ec);
  llvm::sys::fs::directory_iterator end;
  for (; !ec && it != end; it.increment(ec)) {
    llvm::SmallString<256> candidate(it->path());
    llvm::sys::path::append(candidate, "bin", "gcc-ld", "ld.lld");
    if (llvm::sys::fs::exists(candidate)) {
      return std::string(candidate);
    }
  }
  return {};
}

std::string RiscvBackend::findLld() {
  if (const char *override_path = std::getenv("TEENYC_LLD_PATH")) {
    if (llvm::sys::fs::exists(override_path)) {
      return override_path;
    }
  }
  // Prefer the toolchain's own bundled linker over anything on PATH (e.g. a
  // separately apt-installed `lld` package) -- this is what "just works"
  // out of the box on any machine with only teenyc/rustup installed, no
  // extra system dependency required.
  if (std::string toolchainLld = findToolchainLld(); !toolchainLld.empty()) {
    return toolchainLld;
  }
  if (auto found = llvm::sys::findProgramByName("ld.lld")) {
    return *found;
  }
  return {};
}

} // namespace triton
} // namespace mlir
