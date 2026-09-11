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

//! End-to-end coverage for `rustc_codegen_llvm::mlir::target::resolve`:
//! `--target`'s architecture selects the Triton backend (Cuda/Riscv), and
//! `-C target-cpu` is validated per backend instead of being silently
//! reinterpreted (see teenyc-j3a).

#![feature(rustc_private)]

use std::env;
use std::panic::{self, AssertUnwindSafe};
use std::path::{Path, PathBuf};

use rustc_driver::{Callbacks, Compilation, run_compiler};
use rustc_interface::interface;

struct MlirBackendCallbacks;

impl Callbacks for MlirBackendCallbacks {
    fn config(&mut self, config: &mut interface::Config) {
        config.make_codegen_backend = Some(Box::new(|_sess: &rustc_session::Session| {
            rustc_codegen_llvm::mlir::MlirCodegenBackend::new()
        }));
    }
}

/// Selects the `mlir` backend, records the target features rustc derived from
/// its `target_config` once the session exists, and stops before codegen.
#[derive(Default)]
struct TargetFeatureCallbacks {
    features: Vec<String>,
}

impl Callbacks for TargetFeatureCallbacks {
    fn config(&mut self, config: &mut interface::Config) {
        MlirBackendCallbacks.config(config);
    }

    fn after_crate_root_parsing(
        &mut self,
        compiler: &interface::Compiler,
        _krate: &mut rustc_ast::Crate,
    ) -> Compilation {
        self.features =
            compiler.sess.internal_target_features.iter().map(|f| f.to_string()).collect();
        Compilation::Stop
    }
}

/// Compiles `filename` for `target`, with `extra_args` appended (e.g.
/// `-C target-cpu=...`). Returns `Err` if compilation panicked (which is how
/// `sess.dcx().fatal(..)` surfaces through `run_compiler` when used as a
/// library rather than through the `rustc` binary's own exit-code wrapper).
fn try_compile(filename: &Path, target: &str, output_name: &str, extra_args: &[&str]) -> Result<(), String> {
    try_compile_with(&mut MlirBackendCallbacks, filename, target, output_name, extra_args)
}

/// Like [`try_compile`], but driven by `callbacks`, which must select the
/// `mlir` backend themselves (e.g. by delegating to [`MlirBackendCallbacks`]).
fn try_compile_with(
    callbacks: &mut (dyn Callbacks + Send),
    filename: &Path,
    target: &str,
    output_name: &str,
    extra_args: &[&str],
) -> Result<(), String> {
    let output_path = PathBuf::from("/tmp").join(format!("kernel-{output_name}.asm"));

    let mut args = vec![
        "/home/arshadm/.cargo/bin/rustc".to_string(),
        filename.display().to_string(),
        "-Copt-level=3".to_string(),
        "-Cpanic=abort".to_string(),
        format!("-o{}", output_path.display()),
        format!("--target={target}"),
        "--crate-type=lib".to_string(),
        "-C".to_string(),
        "overflow-checks=off".to_string(),
        "--frontend=triton".to_string(),
    ];
    args.extend(extra_args.iter().map(|s| s.to_string()));

    panic::catch_unwind(AssertUnwindSafe(|| run_compiler(&args, callbacks)))
        .map_err(|_| "compilation panicked".to_string())
}

/// The Rust target features rustc records for `target` with `extra_args`.
fn reported_target_features(target: &str, output_name: &str, extra_args: &[&str]) -> Vec<String> {
    let mut callbacks = TargetFeatureCallbacks::default();
    let src = data_file("triton_relu.rs");
    let result = try_compile_with(&mut callbacks, &src, target, output_name, extra_args);
    assert!(result.is_ok(), "expected {target} to get as far as parsing: {result:?}");
    callbacks.features
}

fn data_file(name: &str) -> PathBuf {
    env::current_dir().unwrap().join("tests/data").join(name)
}

#[test]
fn cuda_valid_target_cpu_succeeds() {
    let src = data_file("triton_relu.rs");
    let result = try_compile(&src, "nvptx64-nvidia-cuda", "cuda_valid_cpu", &[
        "-C",
        "target-cpu=sm_90",
    ]);
    assert!(result.is_ok(), "expected sm_90 to be accepted: {result:?}");
}

/// Compiled PTX without the lines that are expected to differ between
/// capabilities: the declared PTX ISA version and the target SM.
fn ptx_without_version_and_target(ptx: &str) -> String {
    ptx.lines()
        .filter(|line| !line.starts_with(".version ") && !line.starts_with(".target "))
        .collect::<Vec<_>>()
        .join("\n")
}

#[test]
fn cuda_capability_107_compiles_as_sm_100() {
    // LLVM's NVPTX backend has no sm_107 (Jetson Thor): given `sm_107a` it
    // ignores the processor and compiles for a generic subtarget. Upstream
    // Triton compiles capability 107 as sm_100 and only stamps `.target sm_107`
    // onto the PTX afterwards.
    let src = data_file("triton_relu.rs");
    let mut ptx = Vec::new();
    for cpu in ["sm_100", "sm_107"] {
        let output_name = format!("cuda_{cpu}");
        let target_cpu = format!("target-cpu={cpu}");
        let result = try_compile(&src, "nvptx64-nvidia-cuda", &output_name, &["-C", &target_cpu]);
        assert!(result.is_ok(), "expected {cpu} to compile: {result:?}");
        ptx.push(
            std::fs::read_to_string(format!("/tmp/kernel-{output_name}.asm"))
                .expect("read compiled PTX"),
        );
    }

    assert!(
        ptx[1].lines().any(|line| line == ".target sm_107a"),
        "capability 107 PTX should still declare its own target"
    );
    assert_eq!(
        ptx_without_version_and_target(&ptx[0]),
        ptx_without_version_and_target(&ptx[1]),
        "capability 107 should be lowered exactly like sm_100"
    );
}

#[test]
fn cuda_default_ptx_versions_meet_llvm_minimums() {
    // CudaBackend passes the resolved PTX version to LLVM as `+ptx<version>`,
    // and LLVM aborts the whole process when that is below the minimum for the
    // SM it compiles for (getMinPTXVersionForSM in NVPTXSubtarget.cpp), so each
    // capability's default version must meet it.
    let src = data_file("triton_relu.rs");
    for capability in [75, 80, 86, 87, 88, 89, 90, 100, 101, 103, 107, 110, 120, 121] {
        let output_name = format!("cuda_ptx_sm_{capability}");
        let target_cpu = format!("target-cpu=sm_{capability}");
        let result = try_compile(&src, "nvptx64-nvidia-cuda", &output_name, &["-C", &target_cpu]);
        assert!(result.is_ok(), "expected sm_{capability} to compile: {result:?}");
    }
}

#[test]
fn cuda_invalid_target_cpu_is_rejected() {
    // Before teenyc-j3a, an unrecognized -C target-cpu silently defaulted to
    // capability 90 instead of being rejected. A RISC-V-shaped cpu string is
    // the sharpest case: it must never be misread as a CUDA capability.
    let src = data_file("triton_relu.rs");
    let result = try_compile(&src, "nvptx64-nvidia-cuda", "cuda_invalid_cpu", &[
        "-C",
        "target-cpu=generic-rvv1.0",
    ]);
    assert!(result.is_err(), "expected an invalid CUDA target-cpu to be rejected");
}

#[test]
fn riscv_target_lowers_relu_kernel_to_assembly() {
    // riscv64-generic selects TargetBackend::Riscv (see
    // rustc_target::spec::targets::riscv64_generic and
    // rustc_codegen_llvm::mlir::target::resolve), which lowers the kernel
    // through the TritonCPU pipeline in CpuBackend. --emit=asm makes the
    // backend hand back RiscvBackend's assembly instead of the linked .so, so
    // the snapshot records what the kernel was actually lowered to.
    //
    // The kernel's block size is kept small: codegen time and output size grow
    // with it (see teenyc-trp). After an intended change, a failing run leaves
    // a pending snapshot; review it from compiler/rustc_codegen_llvm with
    // `cargo insta review`.
    let src = data_file("triton_relu.rs");
    let result = try_compile(&src, "riscv64-generic", "riscv_relu", &["--emit=asm"]);
    assert!(result.is_ok(), "expected the RISC-V relu kernel to compile: {result:?}");

    let bytes = std::fs::read("/tmp/kernel-riscv_relu.asm").expect("read compiled output");
    let asm = String::from_utf8(bytes).expect("--emit=asm output is not assembly text");

    // Debug info records the directory of the source file, which is absolute
    // here and so differs between checkouts.
    let data_dir = src.parent().expect("data file has a parent directory");
    let asm = asm.replace(&*data_dir.to_string_lossy(), "$TEST_DATA");

    insta::assert_snapshot!("riscv64_relu_asm", asm);
}

#[test]
fn riscv_target_reports_its_spec_target_features() {
    // rustc warns (and will eventually error) when a feature the target's ABI
    // requires -- `d` for riscv64-generic's lp64d -- is missing from the
    // features the codegen backend reports. The spec enables m/a/f/d/c/v; the
    // reported set must include them and the features they imply.
    let features = reported_target_features("riscv64-generic", "riscv_features", &[]);
    for feature in ["m", "a", "f", "d", "c", "v", "zve64d", "zvl128b"] {
        assert!(
            features.iter().any(|f| f == feature),
            "`{feature}` missing from reported target features {features:?}"
        );
    }
}

#[test]
fn riscv_target_feature_flag_overrides_spec_features() {
    let features =
        reported_target_features("riscv64-generic", "riscv_no_v", &["-C", "target-feature=-v"]);
    assert!(!features.iter().any(|f| f == "v"), "`-v` left `v` enabled: {features:?}");
    assert!(features.iter().any(|f| f == "d"), "`-v` removed `d`: {features:?}");
}
