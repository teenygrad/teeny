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
    compile_to_mlir_with(file, target, output_name, &[])
}

/// Compiles `file` for `target` and returns its Triton IR: the kernel after the Triton backend
/// has structured its control flow (Backend::structureControlFlow). The frontend lowers every
/// loop as plain branches, so loop structure is checked here (teenyc-3a1).
fn compile_to_ttir(file: &str, target: &str, output_name: &str) -> String {
    compile_to_mlir(file, target, output_name);
    std::fs::read_to_string(format!("/tmp/kernel-{output_name}.ttir"))
        .unwrap_or_else(|e| panic!("read the TTIR for {file} on {target}: {e}"))
}

fn compile_to_mlir_with(file: &str, target: &str, output_name: &str, args: &[&str]) -> String {
    let args: Vec<&str> = ["--emit=asm"].into_iter().chain(args.iter().copied()).collect();
    let result = try_compile(&data_file(file), target, output_name, &args);
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
        let mlir =
            compile_to_ttir("triton_back_to_back_loops.rs", target, &format!("{name}_b2b"));
        assert_eq!(loop_nesting(&mlir).len(), 2, "expected two loops on {target}:\n{mlir}");
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
            compile_to_ttir("triton_range_bounds.rs", target, &format!("{name}_range_bounds"));
        assert_eq!(scf_for_bounds(&mlir), expected, "scf.for bounds on {target} in:\n{mlir}");
    }
}

#[test]
fn counted_while_loop_keeps_its_step_for_every_target() {
    // A hand-written `while i < n { ..; i += 2 }` has the same `counter < end` header as an
    // inlined `Range::next`, minus the `Option`. Lowering it as a unit-step loop would run it once
    // per value below n instead of once per step (teenyc-3jo), so if it becomes an `scf.for`, its
    // step must be 2.
    for (target, name) in LOOP_TARGETS {
        let mlir = compile_to_ttir("triton_counted_while.rs", target, &format!("{name}_while"));
        let loops: Vec<&str> =
            mlir.lines().filter(|l| l.contains("scf.for ") || l.contains("scf.while ")).collect();
        assert_eq!(loops.len(), 1, "expected one loop on {target}:\n{mlir}");
        if loops[0].contains("scf.for ") {
            assert!(loops[0].contains(" step %c2_i32"), "loop lost its step on {target}:\n{mlir}");
        }
    }
}

/// Each `scf.for` / `scf.while` in `mlir` as `(op, depth)`, in order, where depth counts the
/// regions it sits inside (from the MLIR's two-space indentation, the function body being at
/// four).
fn loop_nesting(mlir: &str) -> Vec<(&'static str, usize)> {
    mlir.lines()
        .filter_map(|line| {
            let trimmed = line.trim_start();
            let op = ["scf.for", "scf.while"].into_iter().find(|op| {
                trimmed.starts_with(&format!("{op} ")) || line.contains(&format!(" = {op} "))
            })?;
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
        let mlir =
            compile_to_ttir("triton_nested_loops.rs", target, &format!("{name}_nested"));
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
        let mlir =
            compile_to_ttir("triton_nested_loops_deep.rs", target, &format!("{name}_deep"));
        let depths: Vec<usize> = loop_nesting(&mlir).into_iter().map(|(_, depth)| depth).collect();
        assert_eq!(depths, [0, 1, 2], "expected three nested loops on {target}:\n{mlir}");
    }
}

/// The step of each `scf.for` in `mlir`, in order.
fn scf_for_steps(mlir: &str) -> Vec<String> {
    mlir.lines()
        .filter_map(|line| {
            let rest = line.split_once("scf.for ")?.1.split_once(" step ")?.1;
            Some(rest.split_whitespace().next()?.to_string())
        })
        .collect()
}

#[test]
fn while_loop_around_a_for_loop_lowers_as_nested_loops_for_every_target() {
    // The `while i < n { ..; i += 2 }` is counted, so the backend uplifts it to an scf.for with
    // its step of 2, around the inner `for`.
    for (target, name) in LOOP_TARGETS {
        let mlir =
            compile_to_ttir("triton_while_around_for.rs", target, &format!("{name}_while_for"));
        assert_eq!(
            loop_nesting(&mlir),
            [("scf.for", 0), ("scf.for", 1)],
            "expected an scf.for nested in the while loop on {target}:\n{mlir}"
        );
        assert_eq!(scf_for_steps(&mlir), ["%c2_i32", "%c1_i32"], "loop steps on {target}:\n{mlir}");
    }
}

#[test]
fn loops_inside_or_after_an_if_in_a_loop_lower_as_nested_loops_for_every_target() {
    // teenyc-9nm: an `if` arm in a loop body was lowered as a straight chain of blocks, which
    // couldn't hold a loop, so the inner loop (in the arm, or after the `if`, whose join each arm
    // runs into) was rejected.
    for (target, name) in LOOP_TARGETS {
        let mlir =
            compile_to_ttir("triton_loop_in_if_in_loop.rs", target, &format!("{name}_loop_in_if"));
        let depths: Vec<usize> = loop_nesting(&mlir).into_iter().map(|(_, depth)| depth).collect();
        // The first nest: the inner loop inside the scf.if, one level deeper again.
        assert_eq!(depths.first(), Some(&0), "no outer loop on {target}:\n{mlir}");
        assert!(
            depths.contains(&2),
            "the loop in the `if` isn't nested in the scf.if in the loop on {target}:\n{mlir}"
        );
        // Every inner loop runs over `0..%arg1` (b), not a single iteration.
        let inner_loops =
            mlir.lines().filter(|l| l.contains(" = scf.for ") && l.contains(" to %arg1 ")).count();
        assert!(inner_loops >= 2, "expected both inner loops over b on {target}:\n{mlir}");
    }
}

#[test]
fn loop_inside_an_if_inside_a_while_loop_lowers_for_every_target() {
    for (target, name) in LOOP_TARGETS {
        let mlir = compile_to_ttir(
            "triton_loop_in_if_in_while.rs",
            target,
            &format!("{name}_loop_in_if_in_while"),
        );
        // The inner loop sits in an scf.if arm, two levels inside the while loop, which is
        // counted (`i += 2`) and so uplifted to an scf.for with that step.
        assert_eq!(
            loop_nesting(&mlir),
            [("scf.for", 0), ("scf.for", 2)],
            "expected an scf.for in an scf.if in the while loop on {target}:\n{mlir}"
        );
        assert_eq!(scf_for_steps(&mlir), ["%c2_i32", "%c1_i32"], "loop steps on {target}:\n{mlir}");
    }
}

#[test]
fn nested_ifs_in_a_loop_lower_as_nested_scf_ifs_for_every_target() {
    // teenyc-2uq: an `if` inside an `if` arm in a loop body was ignored, the rest of the body was
    // lowered after the loop's yield, and the loop was folded away: the kernel stored zeros.
    for (target, name) in LOOP_TARGETS {
        let mlir =
            compile_to_ttir("triton_nested_if_in_loop.rs", target, &format!("{name}_nested_if"));
        assert_eq!(loop_nesting(&mlir), [("scf.for", 0)], "no loop on {target}:\n{mlir}");
        assert_store_is_computed(&mlir, target);

        // An if/else with code after it, and a loop in an inner `if`.
        let mlir = compile_to_ttir(
            "triton_nested_ifs_in_loop.rs",
            target,
            &format!("{name}_nested_ifs"),
        );
        let loops = loop_nesting(&mlir);
        assert_eq!(loops.first(), Some(&("scf.for", 0)), "no outer loop on {target}:\n{mlir}");
        assert!(loops.len() > 1, "the inner loop was lost on {target}:\n{mlir}");
        // The inner loop sits in the innermost `if`, inside the outer two.
        assert!(
            loops[1..].iter().all(|&(op, depth)| op == "scf.for" && depth >= 3),
            "expected the inner loop in the nested scf.ifs on {target}:\n{mlir}"
        );
        assert_store_is_computed(&mlir, target);
    }
}

#[test]
fn do_while_loop_carries_its_accumulator() {
    // teenyc-3a1: a `loop { ..; if cond { break } }` isn't a `for` or `while` loop, so it is
    // lowered as plain branches. Its header merged only the counter: the accumulator assigned in
    // the loop got no block argument, so each iteration restarted from zeros. SSA construction
    // gives every local assigned in the loop and live at its header a block argument there.
    let mlir = compile_to_mlir("triton_do_while_loop.rs", "nvptx64-nvidia-cuda", "do-while-cuda");
    let header = mlir.lines().find(|l| l.trim_start().starts_with("^bb1(")).unwrap_or_default();
    assert!(
        header.contains("tensor<16xf32>"),
        "the loop header doesn't carry the accumulator:\n{mlir}"
    );

    // TritonCPU only lowers tensors inside structured control flow, so it crashed on the tensor
    // carried through plain branches until Backend::structureControlFlow lifted them to scf.
    let mlir = compile_to_mlir("triton_do_while_loop.rs", "riscv64-generic", "do-while-riscv");
    assert_store_is_computed(&mlir, "riscv64-generic");
}

#[test]
fn every_loop_compiles_when_lowered_as_plain_branches() {
    // teenyc-3a1: the frontend matches no loop as a `for` or `while`; every loop is lowered to
    // plain branches and Backend::structureControlFlow lifts them back to scf in the Triton
    // backend. Each loop shape must compile through the whole pipeline, to assembly, on every
    // target.
    let files = [
        "triton_back_to_back_loops.rs",
        "triton_counted_while.rs",
        "triton_do_while_loop.rs",
        "triton_loop_in_if_in_loop.rs",
        "triton_loop_in_if_in_while.rs",
        "triton_nested_if_in_loop.rs",
        "triton_nested_ifs_in_loop.rs",
        "triton_nested_loops.rs",
        "triton_nested_loops_deep.rs",
        "triton_option_loop.rs",
        "triton_range_bounds.rs",
        "triton_while_around_for.rs",
    ];
    for file in files {
        for target in ["nvptx64-nvidia-cuda", "riscv64-generic"] {
            let name = format!("plain-branches-{}-{target}", file.trim_end_matches(".rs"));
            let result = try_compile(&data_file(file), target, &name, &["--emit=asm"]);
            assert!(result.is_ok(), "expected {file} to compile for {target}: {result:?}");
            let mlir = std::fs::read_to_string(format!("/tmp/kernel-{name}.mlir"))
                .unwrap_or_else(|e| panic!("read the MLIR for {file} on {target}: {e}"));
            assert!(
                !mlir.contains("scf.for") && !mlir.contains("scf.while"),
                "expected {file} to reach the backend as plain branches on {target}:\n{mlir}"
            );
        }
    }
}

#[test]
fn a_loop_counter_keeps_its_exit_value_for_every_target() {
    // teenyc-3a1: uplifting a lifted `scf.while` to `scf.for` (scf::upliftWhileToForLoop)
    // rebuilt the counter's value after the loop as that of the last iteration, one step short.
    // teeny-kernels' maxpool2d reads the row its skip loop stops at, and pooled the wrong rows.
    // The kernel stores `i` after `while i < 7 { i += 1 }`, which folds to a constant.
    for (target, seven) in [("nvptx64-nvidia-cuda", "mov.b32"), ("riscv64-generic", "vmv.v.i")] {
        let name = format!("loop-counter-exit-{target}");
        compile_to_mlir("triton_loop_counter_exit.rs", target, &name);
        let asm = std::fs::read_to_string(format!("/tmp/kernel-{name}.asm"))
            .unwrap_or_else(|e| panic!("read the assembly for {target}: {e}"));
        let stored: Vec<&str> = asm.lines().filter(|l| l.trim_start().starts_with(seven)).collect();
        assert!(
            stored.iter().any(|l| l.trim_end().ends_with(", 7;") || l.trim_end().ends_with(", 7")),
            "expected the kernel to store 7 on {target}, found {stored:?}:\n{asm}"
        );
    }
}

/// The kernel stores a value it computed (a loop's result, or one merged from both arms of an
/// `if`), not a constant (`%cst`).
fn assert_store_is_computed(mlir: &str, target: &str) {
    let store = mlir.lines().find(|l| l.contains("tt.store")).unwrap_or_default();
    assert!(!store.contains("%cst"), "the kernel stores a constant on {target}:\n{mlir}");
}

#[test]
fn values_chosen_by_an_if_merge_for_every_target() {
    // teenyc-3a1: an Option or tuple local assigned on both arms of an `if` was rejected ("is
    // assigned in more than one place"), since only plain values were merged where paths meet.
    // An Option now merges as an `i1` "is Some" flag and its payload, a tuple field by field.
    // And a local assigned a different constant on each arm was folded to one of them.
    let cases = [
        ("triton_option_join.rs", "option-join"),
        ("triton_tuple_join.rs", "tuple-join"),
        ("triton_option_loop.rs", "option-loop"),
        ("triton_constant_chosen_by_if.rs", "constant-chosen-by-if"),
    ];
    for (file, name) in cases {
        for target in ["nvptx64-nvidia-cuda", "riscv64-generic"] {
            let mlir = compile_to_mlir(file, target, &format!("{name}-{target}"));
            assert_store_is_computed(&mlir, target);
        }
    }
}

#[test]
fn an_option_chosen_at_run_time_is_rejected_as_an_optional_operand() {
    // Triton's optional operands (here a load mask) must be present or absent at compile time,
    // so a mask that is `Some` on one path and `None` on the other is an error, not a guess.
    let result = try_compile(
        &data_file("triton_merged_option_mask.rs"),
        "nvptx64-nvidia-cuda",
        "merged-option-mask",
        &["--emit=asm"],
    );
    assert!(result.is_err(), "expected a mask chosen at run time to be rejected");
}
