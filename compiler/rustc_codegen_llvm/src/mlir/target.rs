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

//! Backend/target-cpu resolution for the MLIR/Triton codegen backend.
//!
//! Follows the same split rustc uses everywhere else: the `--target`
//! triple's *architecture* selects which hardware backend a compile uses
//! (see `nvptx64-nvidia-cuda` and `riscv64-generic` in
//! `rustc_target::spec::targets`, both of which set
//! `default_codegen_backend: Some("mlir")`), and `-C target-cpu` selects the
//! specific model/capability within that backend. `-C target-cpu` values are
//! never silently reinterpreted across backends -- an unrecognized value is
//! a hard error, not a guessed default.

use std::ffi::CString;

use rustc_mlir::ffi::{CompileOptions, OptionalI32};
use rustc_session::Session;
use rustc_target::spec::Arch;

use crate::mlir::module::resolve_ptx_version;

/// Resolves the `CompileOptions` (and Triton `target` name) to use for this
/// session, from `sess.target.arch` and `-C target-cpu`.
///
/// Returns the options, the target-name string `TritonCompiler::new` expects
/// (e.g. `"cuda"`), and any owned C strings the caller must keep alive for
/// the duration of that call -- `RiscvCompileOptions`'s pointer fields borrow
/// from these.
pub(crate) fn resolve(sess: &Session) -> (CompileOptions, &'static str, Vec<CString>) {
    match &sess.target.arch {
        Arch::Nvptx64 => {
            let options = resolve_cuda(sess);
            (options, "cuda", Vec::new())
        }
        Arch::RiscV32 | Arch::RiscV64 => {
            let (options, keep_alive) = resolve_riscv(sess);
            (options, "riscv", keep_alive)
        }
        other => sess.dcx().fatal(format!(
            "the `mlir` codegen backend does not support target architecture `{}`; supported \
             architectures are `nvptx64` (via the `nvptx64-nvidia-cuda` target) and \
             `riscv32`/`riscv64` (via the `riscv64-generic` target)",
            other.desc(),
        )),
    }
}

fn resolve_cuda(sess: &Session) -> CompileOptions {
    let cpu = crate::llvm_util::target_cpu(sess);
    let capability = parse_cuda_capability(cpu).unwrap_or_else(|msg| sess.dcx().fatal(msg));

    let mut options = CompileOptions::default_cuda();
    // Safety: CompileOptionsData is a union; default_cuda() sets the cuda variant.
    options.data.cuda.capability = capability;
    options.data.cuda.ptx_version = OptionalI32::some(resolve_ptx_version(capability));
    // `debug` gates the C++ backend's per-pass IR printing (see
    // CudaBackend::makeTTIR/makeTTGIR/makeLLIR) -- only worth paying for
    // when a subscriber is actually listening at trace level for this
    // backend's log target.
    options.data.cuda.debug =
        tracing::enabled!(target: crate::mlir::LOG_TARGET, tracing::Level::TRACE);

    options
}

/// Parses an NVIDIA SM version out of a `-C target-cpu` value, e.g.
/// `"sm_90"` or `"sm_90a"` -> `90`.
///
/// Unlike LLVM's real `-mcpu` handling (an open-ended, LLVM-version-dependent
/// name list), this backend only understands the `sm_<NN>[a]` shape used
/// throughout `CudaBackend`/`default_ptx_version_for_capability`. Anything
/// else -- including a RISC-V target-cpu string like `spacemit-k3` that
/// reached this function because of some other bug -- is almost certainly
/// meant for a different backend, so this fails loudly instead of silently
/// falling back to a guessed capability.
fn parse_cuda_capability(cpu: &str) -> Result<i32, String> {
    let digits = cpu.strip_prefix("sm_").ok_or_else(|| invalid_cuda_cpu(cpu))?;
    let digits = digits.strip_suffix('a').unwrap_or(digits);
    digits.parse::<i32>().map_err(|_| invalid_cuda_cpu(cpu))
}

fn invalid_cuda_cpu(cpu: &str) -> String {
    format!(
        "invalid `-C target-cpu` value `{cpu}` for the `mlir` codegen backend's CUDA target: \
         expected an NVIDIA SM version in the form `sm_<NN>` or `sm_<NN>a` (e.g. `sm_90`, `sm_90a`)"
    )
}

fn resolve_riscv(sess: &Session) -> (CompileOptions, Vec<CString>) {
    let cpu = crate::llvm_util::target_cpu(sess);
    if cpu.is_empty() {
        sess.dcx().fatal(
            "the `mlir` codegen backend's RISC-V target requires a non-empty `-C target-cpu` \
             (e.g. `spacemit-k3`, `generic-rvv1.0`)",
        );
    }
    let cpu_c = CString::new(cpu).unwrap_or_else(|_| {
        sess.dcx().fatal(format!("`-C target-cpu` value `{cpu}` contains a NUL byte"))
    });
    // `sess.target.llvm_target` (e.g. "riscv64"), not `sess.opts.target_triple`
    // (the rustc-level tuple, e.g. "riscv64-generic") -- the latter isn't a
    // real LLVM triple `TargetRegistry::lookupTarget` can resolve.
    let triple_c = CString::new(sess.target.llvm_target.as_ref()).ok();
    // Built the same way as the LLVM backend's target machine features: the
    // target spec's (riscv64-generic enables `+v`), then `-C target-feature`.
    // Without them LLVM has no vector extension to lower TritonCPU's
    // fixed-width vector ops to, and expands them element by element.
    let features_c = CString::new(crate::llvm_util::global_llvm_features(sess, false).join(","))
        .unwrap_or_else(|_| sess.dcx().fatal("RISC-V target features contain a NUL byte"));

    let mut options = CompileOptions::default_riscv();
    // Safety: CompileOptionsData is a union; default_riscv() sets the riscv variant.
    options.data.riscv.cpu = cpu_c.as_ptr();
    options.data.riscv.features = features_c.as_ptr();
    if let Some(triple_c) = &triple_c {
        options.data.riscv.target_triple = triple_c.as_ptr();
    }
    options.data.riscv.debug =
        tracing::enabled!(target: crate::mlir::LOG_TARGET, tracing::Level::TRACE);

    let mut keep_alive = vec![cpu_c, features_c];
    keep_alive.extend(triple_c);
    (options, keep_alive)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_bare_sm_version() {
        assert_eq!(parse_cuda_capability("sm_90"), Ok(90));
    }

    #[test]
    fn parses_a_suffixed_sm_version() {
        assert_eq!(parse_cuda_capability("sm_120a"), Ok(120));
    }

    #[test]
    fn rejects_missing_sm_prefix() {
        assert!(parse_cuda_capability("spacemit-k3").is_err());
    }

    #[test]
    fn rejects_non_numeric_suffix() {
        assert!(parse_cuda_capability("sm_ninety").is_err());
    }

    #[test]
    fn rejects_riscv_cpu_string_instead_of_defaulting() {
        // The bug this whole change fixes: the old `unwrap_or(90)` fallback
        // would have silently "parsed" a RISC-V target-cpu string as CUDA
        // capability 90 instead of rejecting it.
        assert!(parse_cuda_capability("generic-rvv1.0").is_err());
    }
}
