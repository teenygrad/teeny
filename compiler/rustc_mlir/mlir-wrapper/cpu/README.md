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

### 1. Patched-LLVM dependencies (5 files) — the blocker

triton-cpu is built against a *patched* LLVM, and two of its patches are load-
bearing here. Neither is a matter of `LLVM_TARGETS_TO_BUILD`: X86 is already in
our target list (`X86;AMDGPU;NVPTX;RISCV`). These are MLIR **dialect/library**
patches, a separate axis from LLVM codegen targets.

- **`mlir/Dialect/X86/`** — does not exist in LLVM 22.0. The fork consolidates
  AMX under an umbrella `x86::X86Dialect` at `mlir::x86::amx`; upstream has it
  at `mlir/Dialect/AMX/AMXDialect.h` under plain `mlir::amx`, with no umbrella.
  Breaks `TritonCPUToLLVM/TypeConverter.cpp`,
  `TritonCPUTransforms/ConvertDotOp/ConvertDotTo{AMX,Nanokernel}.cpp`, and
  `include/TritonCPU/Registration.h`. Shallow: header path, namespace prefix,
  and the `x86::X86Dialect` registry entry.
- **`populateVectorMultiReduction{Reorder,Flattening,Unrolling}Patterns`** —
  fork additions to MLIR's vector dialect, absent upstream. Breaks
  `TritonCPUToLLVM/LowerMultiReduction.cpp`. Upstream offers only
  `populateVectorMultiReductionLoweringPatterns`, so this one needs real work,
  not a rename.

Either port these to the upstream equivalents, or carry the triton-cpu LLVM
patches into `src/llvm-project`. The first is likely cheaper for AMX, the
second may be unavoidable for the vector patterns.

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
