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
        config.make_codegen_backend = Some(Box::new(|_sess: &rustc_session::EarlySession| {
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

/// Both backends, as `(target, name)` pairs; `name` keeps each compile's output files apart.
const LOOP_TARGETS: [(&str, &str); 2] =
    [("riscv64-generic", "riscv"), ("nvptx64-nvidia-cuda", "cuda")];

/// Compiles `file` for `target` and returns the MLIR the backend emitted for it.
fn compile_to_mlir(file: &str, target: &str, output_name: &str) -> String {
    let result = try_compile(&data_file(file), target, output_name, &["--emit=asm"]);
    assert!(result.is_ok(), "expected {file} to compile for {target}: {result:?}");
    std::fs::read_to_string(format!("/tmp/kernel-{output_name}.mlir"))
        .unwrap_or_else(|e| panic!("read the MLIR for {file} on {target}: {e}"))
}

#[test]
fn back_to_back_loops_lower_for_every_target() {
    // teenygrad#156: when one `for` loop's exit flows straight into the next loop, the block
    // between them can come after the second loop's header in MIR order. The init-block search
    // ignored it, so the second loop was never built and codegen panicked in codegen_goto.
    //
    // teenyc-3jo: both loops are `for _ in ..`, so their bodies never take the value out of
    // `Some`. On nvptx64 `Range::next` inlines into a Some/None-then-switch shape, and the loop
    // detector rejected it for lacking an induction local; the `Option` then had no SSA form.
    for (target, name) in LOOP_TARGETS {
        let mlir = compile_to_mlir("triton_back_to_back_loops.rs", target, &format!("{name}_b2b"));
        // RISC-V's MIR has no `Option` left in these loops, so they lower as scf.while instead.
        let loops = mlir.matches(" = scf.for ").count() + mlir.matches(" = scf.while ").count();
        assert_eq!(loops, 2, "expected two loops on {target}:\n{mlir}");
    }
}

/// `(lower, upper)` operands of each `scf.for` in `mlir`, in order.
fn scf_for_bounds(mlir: &str) -> Vec<(String, String)> {
    mlir.lines()
        .filter_map(|line| {
            let rest = line.split_once("scf.for ")?.1;
            let (_, rest) = rest.split_once(" = ")?;
            let (lower, rest) = rest.split_once(" to ")?;
            let (upper, _) = rest.split_once(" step ")?;
            Some((lower.to_string(), upper.to_string()))
        })
        .collect()
}

#[test]
fn each_loop_runs_over_its_own_range_for_every_target() {
    // teenyc-16j: each scf.for must run over its own loop's range. The lowering started every
    // loop at 0, so `START..n` ran n times instead of n - START. `entry_point(n, s, m)` runs
    // `3..n` and then `s..m`.
    let expected = [("%c3_i32", "%arg0"), ("%arg1", "%arg2")]
        .map(|(lower, upper)| (lower.to_string(), upper.to_string()));
    for (target, name) in LOOP_TARGETS {
        let mlir =
            compile_to_mlir("triton_range_bounds.rs", target, &format!("{name}_range_bounds"));
        assert_eq!(scf_for_bounds(&mlir), expected, "scf.for bounds on {target} in:\n{mlir}");
    }
}

#[test]
fn counted_while_loop_stays_a_while_loop_for_every_target() {
    // A hand-written `while i < n { ..; i += 2 }` has the same `counter < end` header as an
    // inlined `Range::next`, minus the `Option`. Lowering it as a counted scf.for would run it
    // once per value below n instead of once per step, so it must stay an scf.while even though
    // `for _ in ..` loops no longer need an induction local (teenyc-3jo).
    for (target, name) in LOOP_TARGETS {
        let mlir = compile_to_mlir("triton_counted_while.rs", target, &format!("{name}_while"));
        assert!(
            !mlir.contains("scf.for"),
            "while loop became a counted scf.for on {target}:\n{mlir}"
        );
        assert!(mlir.contains("scf.while"), "expected an scf.while on {target}:\n{mlir}");
    }
}

/// Each `scf.for` / `scf.while` in `mlir` as `(op, depth)`, in order, where depth counts the loops
/// it sits inside (from the MLIR's two-space indentation, the function body being at four).
fn loop_nesting(mlir: &str) -> Vec<(&'static str, usize)> {
    mlir.lines()
        .filter_map(|line| {
            let op = ["scf.for", "scf.while"]
                .into_iter()
                .find(|op| line.contains(&format!(" = {op} ")))?;
            let indent = line.len() - line.trim_start().len();
            Some((op, (indent - 4) / 2))
        })
        .collect()
}

#[test]
fn nested_loops_lower_as_nested_loops_for_every_target() {
    // teenygrad#155: a loop nested in another wasn't built as a loop of its own. Its header
    // became an scf.if around one copy of its body, and its back edge was dropped, so it ran once
    // with its index at 0 (and on nvptx64 the whole nest was lost).
    for (target, name) in LOOP_TARGETS {
        let mlir = compile_to_mlir("triton_nested_loops.rs", target, &format!("{name}_nested"));
        assert_eq!(
            loop_nesting(&mlir),
            [("scf.for", 0), ("scf.for", 1)],
            "expected an scf.for nested in an scf.for on {target}:\n{mlir}"
        );
        // The inner loop's induction value must feed the index (`i * b + j`), not a constant 0.
        assert!(!mlir.contains("scf.if"), "inner loop became an scf.if on {target}:\n{mlir}");
    }
}

#[test]
fn deeply_nested_loops_lower_as_nested_loops_for_every_target() {
    // Three levels, with `for _` at the top and bottom. On RISC-V the innermost `for _` has no
    // `Option` left in its MIR and lowers as an scf.while, nested all the same.
    for (target, name) in LOOP_TARGETS {
        let mlir = compile_to_mlir("triton_nested_loops_deep.rs", target, &format!("{name}_deep"));
        let depths: Vec<usize> = loop_nesting(&mlir).into_iter().map(|(_, depth)| depth).collect();
        assert_eq!(depths, [0, 1, 2], "expected three nested loops on {target}:\n{mlir}");
    }
}

#[test]
fn while_loop_around_a_for_loop_stays_a_while_loop_for_every_target() {
    for (target, name) in LOOP_TARGETS {
        let mlir =
            compile_to_mlir("triton_while_around_for.rs", target, &format!("{name}_while_for"));
        assert_eq!(
            loop_nesting(&mlir),
            [("scf.while", 0), ("scf.for", 1)],
            "expected an scf.for nested in an scf.while on {target}:\n{mlir}"
        );
    }
}

#[test]
fn loop_inside_an_if_inside_a_loop_is_rejected_for_every_target() {
    // teenyc-9nm: an `if` arm in a loop body is lowered as a straight chain of blocks, which
    // can't hold a loop. It must fail to compile rather than drop the loop.
    for (target, name) in LOOP_TARGETS {
        let output_name = format!("{name}_loop_in_if");
        let mlir_path = format!("/tmp/kernel-{output_name}.mlir");
        let _ = std::fs::remove_file(&mlir_path);
        let result =
            try_compile(&data_file("triton_loop_in_if_in_loop.rs"), target, &output_name, &[]);
        assert!(result.is_err(), "expected a loop in an if in a loop to be rejected on {target}");
        assert!(
            !std::path::Path::new(&mlir_path).exists(),
            "MLIR was emitted for a loop in an if in a loop on {target}"
        );
    }
}
