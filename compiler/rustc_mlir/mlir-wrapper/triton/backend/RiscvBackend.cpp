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
#include "llvm/ADT/StringRef.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/LegacyPassManager.h"
#include "llvm/IR/Module.h"
#include "llvm/IRReader/IRReader.h"
#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCAsmParser.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
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

RiscvBackend::RiscvBackend(std::string target, RiscvCompileOptions options)
    : CpuBackend(target, options.debug),
      m_target_triple(options.target_triple ? options.target_triple : ""),
      m_cpu(options.cpu ? options.cpu : ""),
      m_features(options.features ? options.features : "") {}

RiscvBackend::~RiscvBackend() {}

void RiscvBackend::loadDialects(MLIRContext &context) {
  // The TritonCPU lowering does not produce RVVDialect ops yet. It is
  // registered so RVV-specific lowering has somewhere to represent RVV
  // concepts (see RVVDialect.td) once that lowering exists.
  DialectRegistry registry;
  registry.insert<mlir::rvv::RVVDialect>();
  context.appendDialectRegistry(registry);

  CpuBackend::loadDialects(context);
}

LogicalResult RiscvBackend::makeASM(MLIRContext &context, ModuleOp module) {
  auto tm = createTargetMachine();
  if (!tm) {
    return failure();
  }

  llvm::LLVMContext llvmContext;
  auto mod = parseStoredLLVMIR(llvmContext);
  if (!mod) {
    return failure();
  }

  llvm::SmallVector<char, 0> asmBuf;
  {
    llvm::raw_svector_ostream os(asmBuf);
    llvm::legacy::PassManager pm;
    if (tm->addPassesToEmitFile(pm, os, nullptr,
                                llvm::CodeGenFileType::AssemblyFile)) {
      llvm::errs() << "RiscvBackend: failed to add passes to emit assembly\n";
      return failure();
    }
    pm.run(*mod);
  }

  m_asm.assign(asmBuf.data(), asmBuf.size());
  return success();
}

LogicalResult RiscvBackend::makeBIN(MLIRContext &context, ModuleOp module) {
  // TritonCompiler::compile only calls makeBIN after makeASM succeeded, so the
  // module has already been through codegen once. Assembling that output is
  // far cheaper than running instruction selection a second time.
  if (m_asm.empty()) {
    llvm::errs() << "RiscvBackend: makeBIN needs the assembly makeASM emits\n";
    return failure();
  }

  auto tm = createTargetMachine();
  if (!tm) {
    return failure();
  }

  llvm::SmallVector<char, 0> objBuf;
  if (failed(assembleObject(*tm, objBuf))) {
    return failure();
  }

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

LogicalResult
RiscvBackend::assembleObject(const llvm::TargetMachine &tm,
                             llvm::SmallVectorImpl<char> &objBuf) const {
  // Build the MC layer from the target machine makeASM emitted with: the
  // assembler rejects instructions, such as RVV's, whose extensions its
  // subtarget lacks.
  const llvm::Target &target = tm.getTarget();
  const llvm::Triple &triple = tm.getTargetTriple();
  const llvm::MCTargetOptions &mcOptions = tm.Options.MCOptions;

  std::unique_ptr<llvm::MCRegisterInfo> mri(target.createMCRegInfo(triple));
  std::unique_ptr<llvm::MCAsmInfo> mai(
      mri ? target.createMCAsmInfo(*mri, triple, mcOptions) : nullptr);
  std::unique_ptr<llvm::MCSubtargetInfo> sti(target.createMCSubtargetInfo(
      triple, tm.getTargetCPU(), tm.getTargetFeatureString()));
  std::unique_ptr<llvm::MCInstrInfo> mcii(target.createMCInstrInfo());
  if (!mri || !mai || !sti || !mcii) {
    llvm::errs() << "RiscvBackend: failed to create the MC target description "
                    "for "
                 << triple.getTriple() << "\n";
    return failure();
  }

  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(
      llvm::MemoryBuffer::getMemBuffer(m_asm, "<riscv-asm>"), llvm::SMLoc());

  llvm::MCContext ctx(triple, *mai, *mri, *sti, &srcMgr);
  std::unique_ptr<llvm::MCObjectFileInfo> mofi(
      target.createMCObjectFileInfo(ctx, tm.isPositionIndependent()));
  ctx.setObjectFileInfo(mofi.get());

  std::unique_ptr<llvm::MCAsmBackend> mab(
      target.createMCAsmBackend(*sti, *mri, mcOptions));
  std::unique_ptr<llvm::MCCodeEmitter> emitter(
      target.createMCCodeEmitter(*mcii, ctx));
  if (!mab || !emitter) {
    llvm::errs() << "RiscvBackend: failed to create the assembler backend\n";
    return failure();
  }

  llvm::raw_svector_ostream os(objBuf);
  std::unique_ptr<llvm::MCObjectWriter> writer = mab->createObjectWriter(os);
  std::unique_ptr<llvm::MCStreamer> streamer(target.createMCObjectStreamer(
      triple, ctx, std::move(mab), std::move(writer), std::move(emitter),
      *sti));

  std::unique_ptr<llvm::MCAsmParser> parser(
      llvm::createMCAsmParser(srcMgr, ctx, *streamer, *mai));
  std::unique_ptr<llvm::MCTargetAsmParser> targetParser(
      target.createMCAsmParser(*sti, *parser, *mcii));
  if (!targetParser) {
    llvm::errs() << "RiscvBackend: " << triple.getTriple()
                 << " has no assembly parser\n";
    return failure();
  }
  parser->setTargetParser(*targetParser);

  // Diagnostics go through srcMgr; Run returns true if there were any errors.
  if (parser->Run(/*NoInitialTextSection=*/false)) {
    llvm::errs() << "RiscvBackend: failed to assemble makeASM's output\n";
    return failure();
  }
  return success();
}

llvm::Triple RiscvBackend::targetTriple() const {
  return llvm::Triple(llvm::Triple::normalize(
      m_target_triple.empty() ? "riscv64" : m_target_triple));
}

std::string RiscvBackend::getTargetArch() const {
  return targetTriple().getArchName().str();
}

std::string RiscvBackend::llvmFeatures() const {
  // rustc passes the target spec's features plus -C target-feature (see
  // resolve_riscv in rustc_codegen_llvm/src/mlir/target.rs); that is where V
  // comes from. The fallback matters because a generic cpu name alone implies
  // no ISA extensions, which defaults codegen to the soft-float ABI
  // (lp64/ilp32) -- incompatible with the hard-float ABI (lp64d/ilp32d)
  // essentially all real RISC-V Linux userspace actually uses.
  if (!m_features.empty()) {
    return m_features;
  }
  return "+m,+a,+f,+d,+c";
}

std::set<std::string> RiscvBackend::getTargetFeatures() const {
  // CpuBackend selects passes by bare feature name, so keep only the
  // extensions the LLVM feature string enables, without their "+".
  const std::string featureString = llvmFeatures();
  llvm::SmallVector<llvm::StringRef, 8> parts;
  llvm::StringRef(featureString).split(parts, ',', /*MaxSplit=*/-1,
                                       /*KeepEmpty=*/false);

  std::set<std::string> features;
  for (llvm::StringRef part : parts) {
    part = part.trim();
    if (part.consume_front("+")) {
      features.insert(part.str());
    }
  }
  return features;
}

std::unique_ptr<llvm::TargetMachine> RiscvBackend::createTargetMachine() {
  llvm::InitializeAllTargets();
  llvm::InitializeAllTargetInfos();
  llvm::InitializeAllTargetMCs();
  llvm::InitializeAllAsmParsers();
  llvm::InitializeAllAsmPrinters();

  llvm::Triple triple = targetTriple();
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
  // RV64 CPU". So this always uses a real, generic LLVM cpu name matching
  // the triple's width, and ignores m_cpu. Extensions such as V come from
  // llvmFeatures() instead; a chip-name-to-LLVM-cpu mapping is still needed
  // before m_cpu can select chip-specific tuning.
  std::string cpu = triple.isArch64Bit() ? "generic-rv64" : "generic-rv32";

  // PIC: makeBIN links the resulting object into a shared library.
  llvm::TargetOptions opts;
  std::unique_ptr<llvm::TargetMachine> tm(target->createTargetMachine(
      triple, cpu, llvmFeatures(), opts, llvm::Reloc::PIC_, std::nullopt,
      llvm::CodeGenOptLevel::Default));
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
