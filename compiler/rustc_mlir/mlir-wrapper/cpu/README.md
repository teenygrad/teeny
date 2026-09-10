# TritonCPU port

Ported from the `triton-cpu` fork's `third_party/cpu/` tree. Tracked by **teenyc-3yu**.

The layout mirrors upstream — `cpu/include` and `cpu/lib` — because the sources
spell their own headers as `#include "cpu/include/..."`, which resolve against
`mlir-wrapper/` in both the source and the build tree. Do not flatten it.

| | |
|---|---|
| `include/Dialect/TritonCPU/IR` | dialect, ops, types, attrs (`.td` + headers) |
| `include/{TritonCPUToLLVM,TritonCPUTransforms,TritonToTritonCPU}` | `Passes.h` / `Passes.td` per pass library |
| `include/ScalarizePass` | `ScalarizeInterface` op interface |
| `include/TritonCPU/Registration.h` | dialect + pass registration entry point |
| `lib/…` | the four libraries: `TritonCPUIR`, `TritonCPUToLLVM`, `TritonCPUTransforms`, `TritonToTritonCPU` |

## Build status

**Not built by default.** Configure with `-DTRITON_CPU_ENABLE=ON` to turn it on;
it does not compile yet. What the CMake port already does:

- `add_triton_library()` → `add_mlir_dialect_library()` in all four `lib/`
  `CMakeLists.txt` (the former is a Triton-internal macro, unavailable here).
  This also registers each library in `MLIR_DIALECT_LIBS`, which `mlir-wrapper`
  already links — no explicit link entry needed.
- `-fno-rtti` per library, for the same reason as `riscv/Dialect/RVV/IR`.
- The `include/` tablegen rules came across unchanged; they were already plain
  `mlir_tablegen()` / `add_public_tablegen_target()`.

## What remains

Measured, not guessed: with `-DTRITON_CPU_ENABLE=ON`, all **17 TableGen outputs
build clean** (the `.td` files have no LLVM 22.0 drift at all) and **33 of 40
objects compile**. Seven fail, in four groups.

### 1. Upstream LLVM version skew (5 files) — the blocker

triton-cpu builds against **stock upstream LLVM**, not a fork: its
`scripts/build-llvm-project.sh` clones `github.com/llvm/llvm-project` and
`reset --hard`s to the hash in `cmake/llvm-info.json`
(`62b7cf9623fc310525f39ed69aaecc318a909731`, main @ 2026-06-01). No patches are
applied anywhere in that script.

Our `src/llvm-project` is LLVM 22.1 on `chore/rustc-1.97.1-pin`, forked from
upstream main at `e9f758a59b2f` (2026-01-13). Both changes below landed upstream
in the ~4.5-month gap, so this is version skew, not a fork divergence — and it
resolves itself whenever the rustc LLVM pin advances past them.

Note this is not a `LLVM_TARGETS_TO_BUILD` question: X86 is already in our
target list (`X86;AMDGPU;NVPTX;RISCV`). These are MLIR dialect/library changes,
a separate axis from LLVM codegen targets.

- **`67ac275fee18` (2026-02-26) "[mlir][x86] Rename x86vector to x86"** — renames
  the `x86vector` dialect to `x86` and nests AMX beneath it, so
  `mlir/Dialect/AMX/AMXDialect.h` + `mlir::amx` became
  `mlir/Dialect/X86/X86Dialect.h` + `mlir::x86::amx`. Breaks
  `TritonCPUToLLVM/TypeConverter.cpp`,
  `TritonCPUTransforms/ConvertDotOp/ConvertDotTo{AMX,Nanokernel}.cpp` and
  `include/TritonCPU/Registration.h`. Being a pure rename, adapting backwards to
  our `mlir::amx` is mechanical.
- **`613a5c555ebf` (2026-03-04) "[mlir][vector] Replace
  OneDimMultiReductionToTwoDim with OneDimMultiReductionToReduction" (#184241)**
  — added `populateVectorMultiReduction{Reorder,Flattening,Unrolling}Patterns`.
  We have only `populateVectorMultiReductionLoweringPatterns`. Breaks
  `TritonCPUToLLVM/LowerMultiReduction.cpp`. Not a rename; needs real work or a
  backport.

Three options per item: adapt the source backwards to our API, backport the
upstream commit into `src/llvm-project`, or wait for the rustc LLVM pin to move.

### 2. Triton API drift (1 file)

We ship Triton 3.7.1; triton-cpu targets ~3.3. `triton/Tools/Sys/GetEnv.h` was
renamed to `.hpp`, breaking `Dialect/TritonCPU/IR/Dialect.cpp`. Expect more of
this once the file compiles past line 12.

### 3. LLVM 22.0 MLIR API drift (2 files)

- `Dialect/TritonCPU/IR/Ops.cpp` — `DotOp::inferReturnTypes` no longer matches
  its declaration; the signature gained a `PropertyRef` parameter.
- `TritonCPUTransforms/OptimizeMasks.cpp` — `DenseTypedElementsAttr` is gone.

### 4. Optional ukernel deps (not yet reached)

`UkernelOpsToOneDNNLLVM.cpp` and `ConvertDotOpToUkernelOps.cpp` need oneDNN;
`UkernelOpsToXSMMLLVM.cpp` needs libxsmm. All three are still listed in the
build on purpose: `Passes.td` registers their passes unconditionally, so
dropping the sources leaves `registerTritonCPU*Passes()` with undefined
references. Trimming them means trimming `Passes.td` as well.

### Out of scope so far

Upstream's `backend/`, `language/`, `runtime/`, `triton_cpu.cc` and `llvm.cc` —
the pybind and runtime layers.
