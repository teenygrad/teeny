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

use std::ffi::{c_char, c_void};

use mlir_sys::{MlirContext, MlirModule, MlirType};

// ---------------------------------------------------------------------------
// FFI-safe helper types (mirrors CudaBackend.h)
// ---------------------------------------------------------------------------

/// An optional 32-bit signed integer (`OptionalI32` in C++).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptionalI32 {
    pub has_value: bool,
    pub value: i32,
}

impl OptionalI32 {
    pub const NONE: Self = Self { has_value: false, value: 0 };

    pub const fn some(value: i32) -> Self {
        Self { has_value: true, value }
    }
}

/// An optional boolean (`OptionalBool` in C++).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct OptionalBool {
    pub has_value: bool,
    pub value: bool,
}

impl OptionalBool {
    pub const NONE: Self = Self { has_value: false, value: false };

    pub const fn some(value: bool) -> Self {
        Self { has_value: true, value }
    }
}

/// A 3-component integer dimension (`Dim3` in C++).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct Dim3 {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

// ---------------------------------------------------------------------------
// Target backend discriminator (mirrors `enum TargetBackend : uint32_t`)
// ---------------------------------------------------------------------------

#[repr(u32)]
#[derive(Debug, Copy, Clone, PartialEq, Eq)]
pub enum TargetBackend {
    Cuda = 0,
    Rocm = 1,
    Riscv = 2,
}

// ---------------------------------------------------------------------------
// Per-backend compile option structs
//
// All `*const c_char` string fields are null-terminated C strings owned by
// the caller; NULL means "use the backend default".
// All `*const *const c_char` array fields are paired with a `usize` length
// and must remain valid for the duration of the compilation call.
// ---------------------------------------------------------------------------

/// FFI-safe compilation options for the CUDA (Triton NVIDIA) backend.
/// Mirrors `CudaCompileOptions` in `CudaBackend.h`.
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct CudaCompileOptions {
    pub num_warps: i32,
    pub num_ctas: i32,
    pub num_stages: i32,
    pub capability: i32,
    pub maxnreg: OptionalI32,
    pub cluster_dims: Dim3,
    pub ptx_version: OptionalI32,
    pub ptx_options: *const c_char, // NULL = not set
    pub ir_override: *const c_char, // NULL = not set

    pub enable_fp_fusion: bool,
    pub launch_cooperative_grid: bool,
    pub launch_pdl: bool,

    pub supported_fp8_dtypes: *const *const c_char,
    pub supported_fp8_dtypes_len: usize,

    pub deprecated_fp8_dot_operand_dtypes: *const *const c_char,
    pub deprecated_fp8_dot_operand_dtypes_len: usize,

    pub default_dot_input_precision: *const c_char, // NULL = "tf32"
    pub allowed_dot_input_precisions: *const *const c_char,
    pub allowed_dot_input_precisions_len: usize,

    pub max_num_imprecise_acc_default: OptionalBool,

    /// Parallel key/value arrays: (name, path) pairs for external libraries.
    pub extern_lib_keys: *const *const c_char,
    pub extern_lib_values: *const *const c_char,
    pub extern_libs_len: usize,

    pub debug: bool,
    pub backend_name: *const c_char, // NULL = "cuda"
    pub sanitize_overflow: bool,
    pub arch: *const c_char, // NULL = not set
    pub dump_enabled: bool,
    pub enable_experimental_consan: bool,
    pub instrumentation: bool,
    pub disable_line_info: bool,
    pub enable_reflect_ftz: bool,
    pub dump_ir_extract_di_local_variables: bool,
}

impl Default for CudaCompileOptions {
    /// Returns a default set of CUDA compile options that mirrors the C++
    /// struct defaults.  All pointer fields are null (meaning "use backend
    /// default"); arrays are empty.
    fn default() -> Self {
        Self {
            num_warps: 4,
            num_ctas: 1,
            num_stages: 3,
            capability: 90, // AXM TODO: Get from device capabilities
            maxnreg: OptionalI32::NONE,
            cluster_dims: Dim3 { x: 1, y: 1, z: 1 },
            ptx_version: OptionalI32::some(87), // AXM TODO: Get from capabilities
            ptx_options: std::ptr::null(),
            ir_override: std::ptr::null(),

            enable_fp_fusion: true,
            launch_cooperative_grid: false,
            launch_pdl: false,

            supported_fp8_dtypes: std::ptr::null(),
            supported_fp8_dtypes_len: 0,

            deprecated_fp8_dot_operand_dtypes: std::ptr::null(),
            deprecated_fp8_dot_operand_dtypes_len: 0,

            default_dot_input_precision: std::ptr::null(), // backend default: "tf32"
            allowed_dot_input_precisions: std::ptr::null(),
            allowed_dot_input_precisions_len: 0,

            max_num_imprecise_acc_default: OptionalBool::NONE,

            extern_lib_keys: std::ptr::null(),
            extern_lib_values: std::ptr::null(),
            extern_libs_len: 0,

            debug: false,
            backend_name: std::ptr::null(), // backend default: "cuda"
            sanitize_overflow: true,
            arch: std::ptr::null(),
            dump_enabled: false,
            enable_experimental_consan: false,
            instrumentation: false,
            disable_line_info: false,
            enable_reflect_ftz: false,
            dump_ir_extract_di_local_variables: false,
        }
    }
}

// ---------------------------------------------------------------------------
// Tagged union of backend option structs (mirrors `CompileOptionsData` in C++)
// ---------------------------------------------------------------------------

/// FFI-safe compilation options for the RISC-V backend.
/// Mirrors `RiscvCompileOptions` in `RiscvBackend.h`.
///
/// `target_triple` selects the LLVM target, `features` is its LLVM target-feature
/// string (NULL falls back to the RV64GC/RV32GC baseline), and `debug` prints
/// the IR after each pass. `cpu` is a Triton-side chip identifier not yet mapped
/// onto an LLVM cpu name, so the target machine uses a generic one (see
/// `RiscvBackend::createTargetMachine`).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct RiscvCompileOptions {
    pub target_triple: *const c_char, // NULL = backend default
    pub cpu: *const c_char,           // NULL = backend default
    pub features: *const c_char,      // NULL = backend default
    pub debug: bool,
}

impl Default for RiscvCompileOptions {
    fn default() -> Self {
        Self {
            target_triple: std::ptr::null(),
            cpu: std::ptr::null(),
            features: std::ptr::null(),
            debug: false,
        }
    }
}

/// Union of all per-backend option structs.  Only the member corresponding to
/// `CompileOptions::backend` may be read.
///
/// # Safety
/// This is a `#[repr(C)]` union.  Accessing the wrong variant is undefined
/// behaviour; always check `CompileOptions::backend` first.
#[repr(C)]
#[derive(Copy, Clone)]
pub union CompileOptionsData {
    pub cuda: CudaCompileOptions,
    pub riscv: RiscvCompileOptions,
    // pub rocm:  RocmCompileOptions,  // reserved for future use
}

/// Complete compile options passed to `mlirTritonCompilerCreate`.
/// Mirrors `struct CompileOptions` in `Compiler.h`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct CompileOptions {
    /// Selects the active member of `data`.
    pub backend: TargetBackend,
    pub data: CompileOptionsData,
}

impl CompileOptions {
    /// Returns a `CompileOptions` populated with default CUDA settings.
    pub fn default_cuda() -> Self {
        Self {
            backend: TargetBackend::Cuda,
            data: CompileOptionsData { cuda: CudaCompileOptions::default() },
        }
    }

    /// Returns a `CompileOptions` populated with default RISC-V settings.
    ///
    /// The RISC-V backend is currently a stub (see `RiscvBackend.h`): every
    /// codegen stage returns `Error::NotImplemented`, so this is only useful
    /// for exercising the FFI plumbing today.
    pub fn default_riscv() -> Self {
        Self {
            backend: TargetBackend::Riscv,
            data: CompileOptionsData { riscv: RiscvCompileOptions::default() },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn target_backend_discriminants_match_cpp() {
        // Must stay in lockstep with `enum TargetBackend : uint32_t` in
        // Compiler.h.
        assert_eq!(TargetBackend::Cuda as u32, 0);
        assert_eq!(TargetBackend::Rocm as u32, 1);
        assert_eq!(TargetBackend::Riscv as u32, 2);
    }

    #[test]
    fn default_riscv_selects_riscv_backend_with_default_options() {
        let options = CompileOptions::default_riscv();
        assert_eq!(options.backend, TargetBackend::Riscv);

        // Safety: `backend` is `Riscv`, so reading the `riscv` union member
        // is the variant selected by the discriminant.
        let riscv = unsafe { options.data.riscv };
        assert!(riscv.target_triple.is_null());
        assert!(riscv.cpu.is_null());
        assert!(riscv.features.is_null());
        assert!(!riscv.debug);
    }
}

// ---------------------------------------------------------------------------
// Opaque handle + extern declarations
// ---------------------------------------------------------------------------

/// Opaque handle to the Triton compiler (C ABI: `struct MlirTritonCompiler { void *ptr; }`).
#[repr(C)]
#[derive(Debug, Copy, Clone)]
pub struct MlirTritonCompiler {
    pub ptr: *mut c_void,
}

#[link(name = "mlir-wrapper", kind = "static")]
unsafe extern "C" {
    pub fn mlirLoadTritonDialect(context: MlirContext);

    pub fn mlirCreateTritonPointerType(pointee: MlirType, address_space: i32) -> MlirType;

    // Triton compiler opaque handle API
    pub fn mlirTritonCompilerCreate(
        context: MlirContext,
        target: *const c_char,
        options: *const CompileOptions,
    ) -> MlirTritonCompiler;

    pub fn mlirTritonCompilerCompile(compiler: MlirTritonCompiler, module: MlirModule) -> bool;

    pub fn mlirTritonCompilerGetLLIR(compiler: MlirTritonCompiler) -> *const c_char;
    pub fn mlirTritonCompilerGetTTIR(compiler: MlirTritonCompiler) -> *const c_char;
    pub fn mlirTritonCompilerGetTTGIR(compiler: MlirTritonCompiler) -> *const c_char;
    pub fn mlirTritonCompilerGetLLVMIR(compiler: MlirTritonCompiler) -> *const c_char;
    pub fn mlirTritonCompilerGetASM(compiler: MlirTritonCompiler) -> *const c_char;
    /// May contain embedded NUL bytes (e.g. a linked ELF shared library) --
    /// always pair with `mlirTritonCompilerGetBINSize` rather than treating
    /// this as a NUL-terminated C string. Returned as `*const u8` (not
    /// `*const c_char`) so the type itself signals this is a raw byte
    /// buffer.
    pub fn mlirTritonCompilerGetBIN(compiler: MlirTritonCompiler) -> *const u8;
    pub fn mlirTritonCompilerGetBINSize(compiler: MlirTritonCompiler) -> usize;

    pub fn mlirTritonCompilerFree(compiler: MlirTritonCompiler);
}
