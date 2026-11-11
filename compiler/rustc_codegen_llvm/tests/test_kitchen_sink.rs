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

#![feature(rustc_private)]

use std::env;
use std::path::{Path, PathBuf};

use rustc_driver::{Callbacks, run_compiler};
use rustc_interface::interface;
use tracing::{debug, info};

/// Custom callbacks that register the MLIR codegen backend programmatically
struct MlirBackendCallbacks;

impl Callbacks for MlirBackendCallbacks {
    fn config(&mut self, config: &mut interface::Config) {
        debug!("MlirBackendCallbacks::config called - registering backend");
        // Register the MLIR codegen backend programmatically
        // This closure will be called when rustc needs to create the codegen backend
        config.make_codegen_backend = Some(Box::new(|_sess: &rustc_session::Session| {
            debug!("make_codegen_backend closure called - creating MlirCodegenBackend");
            // Create and return the MLIR codegen backend
            rustc_codegen_llvm::mlir::MlirCodegenBackend::new()
        }));
    }
}

#[derive(Debug, Clone, Default)]
pub struct LlvmCompiler {}

impl LlvmCompiler {
    pub fn new() -> Self {
        Self {}
    }

    /// Compiles `filename`, writing PTX to `/tmp/kernel-{output_name}.asm` and
    /// returning that path. `output_name` should be unique per call site (e.g.
    /// the test name) so that concurrently-running tests don't clobber each
    /// other's output.
    pub fn compile(
        &self,
        filename: &Path,
        target: &str,
        output_name: &str,
    ) -> Result<PathBuf, Box<dyn std::error::Error>> {
        let working_dir = PathBuf::from("/tmp");

        // Use custom callbacks that register the MLIR backend
        let mut callbacks = MlirBackendCallbacks;
        let exe_name = "/home/arshadm/.cargo/bin/rustc".to_string(); // AXM FIXME: remove this once API changes
        let output_path = working_dir.join(format!("kernel-{output_name}.asm"));
        let output = format!("-o{}", output_path.display());
        let build_type = "-Copt-level=3".to_string(); // Use opt-level=3 for release build
        let panic_abort = "-Cpanic=abort".to_string();
        let target = format!("--target={}", target);
        let crate_type = "--crate-type=lib".to_string();
        // let emit = "--emit=llvm-ir".to_string();
        let overflow_checks = "-C".to_string();
        let overflow_checks_off = "overflow-checks=off".to_string();
        let frontend = "--frontend=triton".to_string();

        info!("Working directory: {}", working_dir.display());
        info!("Target: {}", target);
        info!("Output: {}", output);
        debug!(
            "Rustc command: {} {} {} {} {}",
            exe_name,
            filename.display(),
            output,
            target,
            crate_type
        );

        // Build the arguments for the compiler
        // Note: We no longer need -Zcodegen-backend flag since we're registering
        // the backend programmatically via the callbacks
        let args = vec![
            exe_name,
            filename.display().to_string(),
            build_type,
            panic_abort,
            output,
            target,
            crate_type,
            // emit,
            overflow_checks,
            overflow_checks_off,
            frontend,
        ];

        run_compiler(&args, &mut callbacks);

        Ok(output_path)
    }
}

#[cfg(test)]
mod tests {
    use tracing_subscriber::{EnvFilter, fmt};

    use super::*;

    #[test]
    fn test_triton_kitchen_sink() {
        // Set RUST_LOG=debug in environment to see debug output
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let tensor_add = env::current_dir().unwrap().join("tests/data/triton_kitchen_sink.rs");
        let target = "nvptx64-nvidia-cuda";
        println!("Compiling tensor add with target: {}", tensor_add.display());
        let result = compiler.compile(&tensor_add, target, "test_triton_kitchen_sink");
        assert!(result.is_ok());

        // Verify no unresolved __nv_* externs remain in the PTX — they indicate
        // that libdevice was not linked and the PTX JIT will fail at runtime.
        let ptx = std::fs::read_to_string(result.unwrap()).expect("kernel.asm not written");
        let unresolved: Vec<&str> = ptx
            .lines()
            .filter(|l| l.contains(".extern .func") && l.contains("__nv_"))
            .collect();
        assert!(
            unresolved.is_empty(),
            "PTX contains unresolved __nv_* externs (libdevice not linked):\n{}",
            unresolved.join("\n")
        );

        // teenyc-7lq: an unconditional diverging call once let the MIR inliner reduce the whole
        // kernel to an infinite loop, so the test passed no matter what followed it. The final
        // `x / maximum(s, eps)` and `x / minimum(s, eps)` stores sit at the end of the kernel,
        // so their ops in the PTX show that codegen reached it.
        let has_op = |op: &str, ty: &str| {
            ptx.lines().any(|l| l.trim_start().starts_with(op) && l.contains(ty))
        };
        for (op, ty) in [("ld.global", ""), ("st.global", ""), ("max.", ".f32"), ("min.", ".f32")]
        {
            assert!(has_op(op, ty), "no `{op}` {ty} instruction in PTX:\n{ptx}");
        }
    }

    // Fixed: a compound 4-condition early return inside a Triton kernel previously panicked with
    // "cond_br: phi local not in ssa_values". The fix tightens the phi-join threshold in
    // compute_phi_join_locals from `count >= 2` to `count == preds.len()`, eliminating false
    // phi locals for join blocks with 3+ predecessors.
    // See: compiler/rustc_codegen_llvm/src/mlir/codegen/triton/mod.rs
    #[test]
    fn test_phi_bug_early_return() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir().unwrap().join("tests/data/phi_bug_early_return.rs");
        let result = compiler.compile(&file, "nvptx64-nvidia-cuda", "test_phi_bug_early_return");
        assert!(result.is_ok(), "phi_bug_early_return compile failed: {:?}", result.err());
    }

    // Fixed: a `continue` statement inside a `for` loop in a Triton kernel previously panicked
    // with "codegen_goto: phi local not in ssa_values at branch to bbN". The fix has two parts:
    // (1) collect_body_blocks_ordered now follows SwitchInt successors so the loop is detected,
    // (2) codegen_loop_body_switch_as_scf_if emits scf.if for intra-body SwitchInt targets.
    // See: compiler/rustc_codegen_llvm/src/mlir/codegen/triton/mod.rs and ops/terminator.rs
    //
    // Currently ignored: ops/terminator.rs's codegen_switch_int only handles SwitchInt with
    // zero or one explicit case (`[]` -> unconditional goto, `[(val, target_bb)]` -> cf.cond_br,
    // with codegen_loop_body_switch_as_scf_if special-casing the binary in-loop-body form as
    // scf.if). This kernel's `for i in 0..n { if i >= limit { continue; } ... }` desugars to a
    // SwitchInt on `Iterator::next()`'s Option discriminant with *two* explicit cases (0, 1)
    // plus rustc's default `otherwise` arm (3 targets total) -- the loop's own continue/exit
    // dispatch -- which falls into the `_ => todo!("SwitchInt with {} cases", ...)` catch-all
    // at terminator.rs:880.
    //
    // Generalizing this is real design work, not a mechanical fix, because the switch is
    // loop-carry-aware: `acc` must be threaded through whichever arm is taken. Two directions
    // considered:
    //   1. MLIR's `cf.switch` (already bound in melior/dialect/cf.rs) handles the N-way case
    //      cleanly outside a loop body, but has no built-in notion of loop-carried values, so it
    //      doesn't fit this switch's actual role as the loop's continue/exit dispatch.
    //   2. Extending codegen_loop_body_switch_as_scf_if's hand-built binary scf.if lowering to
    //      N-way (nested scf.if, or scf.index_switch with a per-case yield of `acc`) is the more
    //      likely correct fix, but is new codegen logic in an area this test file's own history
    //      shows is prone to *silent* miscompilation rather than a loud failure when wrong (see
    //      test_float_to_float_scalar_cast's poison-corruption bug above) -- not something to
    //      guess at without deliberately designing and verifying it.
    //
    // Separately, and not yet investigated at all: even a correct MLIR lowering here still has
    // to survive Triton's own MLIR passes (makeTTIR/makeTTGIR), which are already known to be
    // fussy about certain `cf` constructs (see foldDegenerateCondBranches's workaround in
    // Backend.cpp for degenerate cf.cond_br). Whether Triton's pipeline accepts a `cf.switch` or
    // multi-arm `scf.if`/`scf.index_switch` here at all is a separate open question from getting
    // our own MLIR generation right.
    #[ignore = "codegen_switch_int doesn't yet handle SwitchInt with >1 explicit case (N-way, \
                loop-carry-aware lowering needed); see comment above for analysis"]
    #[test]
    fn test_phi_bug_loop_continue() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir().unwrap().join("tests/data/phi_bug_loop_continue.rs");
        let result = compiler.compile(&file, "nvptx64-nvidia-cuda", "test_phi_bug_loop_continue");
        assert!(result.is_ok(), "phi_bug_loop_continue compile failed: {:?}", result.err());
    }

    // Fixed: `codegen_cast` had no arm for `CastKind::FloatToFloat`, so scalar float
    // widen/narrow casts (`x as f64`, `value as f32`) fell into the catch-all and silently
    // became `ub.poison` instead of a real `arith.extf`/`arith.truncf`. This is exactly the
    // pattern generic `Float::from_f64(value: f64) -> Self` impls use, and it corrupted
    // vision-rs's Flash-Attention-2 / PSA kernel constants (online-softmax running max and
    // softmax scale) with no compile-time signal — only detectable by comparing runtime
    // output against expected values. The fix adds a real `FloatToFloat` arm, and turns the
    // remaining unhandled-cast-kind fallback into a hard `MlirError` instead of `ub.poison`,
    // so the next unsupported pattern fails to compile instead of silently miscompiling.
    // See: compiler/rustc_codegen_llvm/src/mlir/codegen/triton/mod.rs (codegen_cast).
    #[test]
    fn test_float_to_float_scalar_cast() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir().unwrap().join("tests/data/float_to_float_scalar_cast.rs");
        let result =
            compiler.compile(&file, "nvptx64-nvidia-cuda", "test_float_to_float_scalar_cast");
        assert!(result.is_ok(), "float_to_float_scalar_cast compile failed: {:?}", result.err());

        // With the bug, compilation *succeeds* — the whole danger was that
        // `Float::from_f64`'s `x as f64` / `value as f32` casts silently became `ub.poison`
        // instead of failing loudly, so `result.is_ok()` alone can't catch a regression.
        // Poison also doesn't reliably leave a textual "poison" trace in the final PTX —
        // register allocation just reuses whatever value happens to land in that slot,
        // which is exactly how this went undetected in vision-rs's Flash-Attention-2 kernel.
        //
        // Instead check that the value actually flows through: `x` round-trips through
        // `D::from_f64(x as f64)` as an identity for `D = f32`, and LLVM folds the resulting
        // (correct) extf/truncf pair away entirely, so the register `entry_point_param_0` is
        // loaded into is the same register passed to the final store. With the bug, that
        // register is never referenced again — the stored value comes from an unrelated
        // poison-allocated register instead.
        let ptx = std::fs::read_to_string(result.unwrap()).expect("kernel.asm not written");
        let param_line = ptx
            .lines()
            .find(|l| l.contains("ld.param.b32") && l.contains("entry_point_param_0"))
            .unwrap_or_else(|| panic!("no ld.param.b32 for entry_point_param_0 in PTX:\n{ptx}"));
        let param_reg = param_line
            .split_whitespace()
            .find(|t| t.starts_with('%'))
            .map(|t| t.trim_end_matches(','))
            .unwrap_or_else(|| panic!("couldn't find register operand in {param_line:?}"));
        assert!(
            ptx.contains(&format!("{{ {param_reg} }}")),
            "register {param_reg} (loaded from entry_point_param_0) is never stored — the \
             FloatToFloat round-trip through Float::from_f64 did not propagate the real \
             value (this is the poison-corruption signature). Full PTX:\n{ptx}"
        );
    }

    // teenyc-o4d / teenyc-3p3: T::maximum and T::minimum with operands of different shapes
    // (a keep-dims reduction against a full tile) used to emit arith.maximumf/minimumf on
    // mismatched types, which failed MLIR verification and panicked the backend.
    #[test]
    fn test_minmax_broadcast() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir().unwrap().join("tests/data/triton_minmax_broadcast.rs");
        let result = compiler.compile(&file, "nvptx64-nvidia-cuda", "test_minmax_broadcast");
        assert!(result.is_ok(), "minmax_broadcast compile failed: {:?}", result.err());

        // Compiling isn't enough on its own: the kernel must still contain both ops, so a
        // regression that optimises them away (as the kitchen sink's dead code does, teenyc-7lq)
        // can't pass silently.
        let ptx = std::fs::read_to_string(result.unwrap()).expect("kernel.asm not written");
        let has_op = |op: &str| {
            ptx.lines().any(|l| l.trim_start().starts_with(op) && l.contains(".f32"))
        };
        assert!(has_op("max."), "no f32 max instruction in PTX:\n{ptx}");
        assert!(has_op("min."), "no f32 min instruction in PTX:\n{ptx}");
    }

    // teenyc-38c: a repeated shape array that MIR GVN turns into `_b = copy _a` used to panic
    // with "Value not found for local", because constant arrays are tracked in `const_arrays`
    // rather than `ssa_values`.
    #[test]
    fn test_const_array_copy() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir().unwrap().join("tests/data/triton_const_array_copy.rs");
        let result = compiler.compile(&file, "nvptx64-nvidia-cuda", "test_const_array_copy");
        assert!(result.is_ok(), "const_array_copy compile failed: {:?}", result.err());

        // The store through the pointers built from the copied shape must survive.
        let ptx = std::fs::read_to_string(result.unwrap()).expect("kernel.asm not written");
        assert!(
            ptx.lines().any(|l| l.trim_start().starts_with("add.") && l.contains(".f32")),
            "no f32 add in PTX:\n{ptx}"
        );
        assert!(
            ptx.lines().any(|l| l.trim_start().starts_with("st.global")),
            "no global store in PTX:\n{ptx}"
        );
    }

    // teenyc-3bk: `T::load` with `other` but no mask used to emit a tt.load that failed
    // verification ("mask type matches ptr type"), panicking the backend.
    #[test]
    fn test_load_other_without_mask() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file =
            env::current_dir().unwrap().join("tests/data/triton_load_other_without_mask.rs");
        let result =
            compiler.compile(&file, "nvptx64-nvidia-cuda", "test_load_other_without_mask");
        assert!(result.is_ok(), "load_other_without_mask compile failed: {:?}", result.err());

        // The load must survive, feeding the add that is stored.
        let ptx = std::fs::read_to_string(result.unwrap()).expect("kernel.asm not written");
        for op in ["ld.global", "add.", "st.global"] {
            assert!(
                ptx.lines().any(|l| l.trim_start().starts_with(op)),
                "no `{op}` instruction in PTX:\n{ptx}"
            );
        }
    }

    // teenyc-37e: `T::reduce` / `T::associative_scan` with a user combine fn used to ICE on the
    // fn-pointer cast, and behind that both ops were stubs that emitted `ub.poison`.
    #[test]
    fn test_reduce_scan_combine() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir().unwrap().join("tests/data/triton_reduce_scan_combine.rs");
        let result = compiler.compile(&file, "nvptx64-nvidia-cuda", "test_reduce_scan_combine");
        assert!(result.is_ok(), "reduce_scan_combine compile failed: {:?}", result.err());

        // The kernel's only multiply is reduce's combine body and its only add is the scan's,
        // so each must appear for its combine region to have been generated (the old stubs
        // produced poison and neither op).
        let ptx = std::fs::read_to_string(result.unwrap()).expect("kernel.asm not written");
        let has_f32_op = |op: &str| {
            ptx.lines().any(|l| l.trim_start().starts_with(op) && l.contains(".f32"))
        };
        assert!(has_f32_op("mul."), "no f32 mul (reduce combine body) in PTX:\n{ptx}");
        assert!(has_f32_op("add."), "no f32 add (scan combine body) in PTX:\n{ptx}");
        assert!(
            ptx.lines().any(|l| l.trim_start().starts_with("st.global")),
            "no global store in PTX:\n{ptx}"
        );
    }

    // teenyc-1jf: `T::make_block_ptr` / `T::advance` emitted `tt.make_tensor_ptr` / `tt.advance`,
    // which Triton 3.6 no longer registers, so the module failed to verify.
    #[test]
    fn test_block_ptr() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir().unwrap().join("tests/data/triton_block_ptr.rs");
        let result = compiler.compile(&file, "nvptx64-nvidia-cuda", "test_block_ptr");
        assert!(result.is_ok(), "block_ptr compile failed: {:?}", result.err());

        // The block pointer has no consumer, so the rest of the kernel must still compile intact.
        let ptx = std::fs::read_to_string(result.unwrap()).expect("kernel.asm not written");
        for op in ["ld.global", "add.", "st.global"] {
            assert!(
                ptx.lines().any(|l| l.trim_start().starts_with(op)),
                "no `{op}` instruction in PTX:\n{ptx}"
            );
        }
    }

    // teenyc-3vx: integer T::maximum emitted arith.maximumf, which fails verification, and
    // T::minimum used the signed op for unsigned tensors.
    #[test]
    fn test_int_minmax() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir().unwrap().join("tests/data/triton_int_minmax.rs");
        let result = compiler.compile(&file, "nvptx64-nvidia-cuda", "test_int_minmax");
        assert!(result.is_ok(), "int_minmax compile failed: {:?}", result.err());

        let ptx = std::fs::read_to_string(result.unwrap()).expect("kernel.asm not written");
        for op in ["max.s32", "min.s32", "max.u32", "min.u32"] {
            assert!(
                ptx.lines().any(|l| l.trim_start().starts_with(op)),
                "no `{op}` instruction in PTX:\n{ptx}"
            );
        }
    }

    // teenyc-1at: argmax / argmin / max_with_indices / min_with_indices were stubs emitting
    // ub.poison, so their results were garbage.
    #[test]
    fn test_reduce_with_indices() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir().unwrap().join("tests/data/triton_reduce_with_indices.rs");
        let result = compiler.compile(&file, "nvptx64-nvidia-cuda", "test_reduce_with_indices");
        assert!(result.is_ok(), "reduce_with_indices compile failed: {:?}", result.err());

        // The combines compare values (gt for max, lt for min), compare indices for max's
        // tie-break, and select both value and index; poison stubs produce none of these.
        let ptx = std::fs::read_to_string(result.unwrap()).expect("kernel.asm not written");
        for op in ["setp.gt.f32", "setp.lt.f32", "setp.lt.s32", "selp.b32", "st.global"] {
            assert!(
                ptx.lines().any(|l| l.trim_start().starts_with(op)),
                "no `{op}` instruction in PTX:\n{ptx}"
            );
        }
    }

    // teenyc-1h2: T::flip was a stub emitting ub.poison.
    #[test]
    fn test_flip() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir().unwrap().join("tests/data/triton_flip.rs");
        let result = compiler.compile(&file, "nvptx64-nvidia-cuda", "test_flip");
        assert!(result.is_ok(), "flip compile failed: {:?}", result.err());

        let ptx = std::fs::read_to_string(result.unwrap()).expect("kernel.asm not written");
        for op in ["ld.global", "st.global"] {
            assert!(
                ptx.lines().any(|l| l.trim_start().starts_with(op)),
                "no `{op}` instruction in PTX:\n{ptx}"
            );
        }
    }

    // teenyc-3er: T::swizzle2d was a stub emitting ub.poison.
    #[test]
    fn test_swizzle2d() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir().unwrap().join("tests/data/triton_swizzle2d.rs");
        let result = compiler.compile(&file, "nvptx64-nvidia-cuda", "test_swizzle2d");
        assert!(result.is_ok(), "swizzle2d compile failed: {:?}", result.err());

        // With runtime inputs, swizzle2d's two divisions (group index, then column within the
        // group) and its `minimum` must survive into the PTX. Its `%`s are not asserted: LLVM
        // rewrites each as `x - (x / y) * y`, reusing the division.
        let ptx = std::fs::read_to_string(result.unwrap()).expect("kernel.asm not written");
        let count = |op: &str| ptx.lines().filter(|l| l.trim_start().starts_with(op)).count();
        assert!(count("div.s32") >= 2, "expected two `div.s32` in PTX:\n{ptx}");
        for op in ["min.s32", "st.global"] {
            assert!(count(op) > 0, "no `{op}` instruction in PTX:\n{ptx}");
        }
    }

    // teenyc-1k4: T::rand / randn / randint / randint4x were stubs emitting ub.poison.
    #[test]
    fn test_philox() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir().unwrap().join("tests/data/triton_philox.rs");
        let result = compiler.compile(&file, "nvptx64-nvidia-cuda", "test_philox");
        assert!(result.is_ok(), "philox compile failed: {:?}", result.err());

        // Philox rounds are high multiplies and xors; rand converts the int to float.
        let ptx = std::fs::read_to_string(result.unwrap()).expect("kernel.asm not written");
        for op in ["mul.hi.u32", "xor.b32", "cvt.rn.f32.s32", "st.global"] {
            assert!(
                ptx.lines().any(|l| l.trim_start().starts_with(op)),
                "no `{op}` instruction in PTX:\n{ptx}"
            );
        }
    }

    // teenyc-190: T::sort was a stub emitting ub.poison.
    #[test]
    fn test_sort() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir().unwrap().join("tests/data/triton_sort.rs");
        let result = compiler.compile(&file, "nvptx64-nvidia-cuda", "test_sort");
        assert!(result.is_ok(), "sort compile failed: {:?}", result.err());

        // Each compare-and-swap compares an element with its partner (`x > y`) and selects.
        let ptx = std::fs::read_to_string(result.unwrap()).expect("kernel.asm not written");
        for op in ["setp.gt.f32", "setp.gt.s32", "selp", "st.global"] {
            assert!(
                ptx.lines().any(|l| l.trim_start().starts_with(op)),
                "no `{op}` instruction in PTX:\n{ptx}"
            );
        }
    }

    // teenyc-3h1: T::inline_asm_elementwise cannot be lowered from its current signature, so it
    // must fail to compile rather than silently producing ub.poison. A codegen error panics
    // inside rustc, and that panic reaches the test through `run_compiler`.
    #[test]
    #[should_panic(expected = "inline_asm_elementwise is not supported")]
    fn test_inline_asm_elementwise_rejected() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir().unwrap().join("tests/data/triton_inline_asm.rs");
        let _ = compiler.compile(&file, "nvptx64-nvidia-cuda", "test_inline_asm_elementwise");
    }

    // teenyc-26l / teenyc-z81: reduction axes (`None`, negative, `Some(k)` on N-D) and N-D
    // keep_dims, plus softmax's dim. A wrong axis or dropped dimension gives a result whose
    // shape cannot be combined with the full-size tensor, which fails MLIR verification.
    #[test]
    fn test_reduce_axis() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir().unwrap().join("tests/data/triton_reduce_axis.rs");
        let result = compiler.compile(&file, "nvptx64-nvidia-cuda", "test_reduce_axis");
        assert!(result.is_ok(), "reduce_axis compile failed: {:?}", result.err());

        let ptx = std::fs::read_to_string(result.unwrap()).expect("kernel.asm not written");
        assert!(
            ptx.lines().any(|l| l.trim_start().starts_with("st.global")),
            "no global store in PTX:\n{ptx}"
        );
    }

    // teenyc-32u: tensor descriptor ops with literal (promoted) slice arguments used to fall
    // back to ub.poison, which silently dropped the copy.
    #[test]
    fn test_tensor_descriptor() {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir().unwrap().join("tests/data/triton_tensor_descriptor.rs");
        let result = compiler.compile(&file, "nvptx64-nvidia-cuda", "test_tensor_descriptor");
        assert!(result.is_ok(), "tensor_descriptor compile failed: {:?}", result.err());

        let ptx = std::fs::read_to_string(result.unwrap()).expect("kernel.asm not written");
        // The tile stores are predicated (`@%p st.global...`).
        assert!(ptx.contains("st.global"), "no global store in PTX:\n{ptx}");
    }

    // teenyc-32u: a descriptor op whose slice argument is chosen at run time must fail to
    // compile; it used to emit ub.poison, or drop the store entirely. Since teenyc-2m7 the slice
    // local is rejected when it is recorded, before the descriptor op. A codegen error panics
    // inside rustc and reaches the test through `run_compiler`.
    fn compile_dyn_descriptor(site: &str) {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file = env::current_dir()
            .unwrap()
            .join(format!("tests/data/triton_tensor_descriptor_dyn_{site}.rs"));
        let _ = compiler.compile(&file, "nvptx64-nvidia-cuda", &format!("test_descriptor_dyn_{site}"));
    }

    #[test]
    #[should_panic(expected = "is assigned in more than one place")]
    fn test_make_tensor_descriptor_dyn_shape_rejected() {
        compile_dyn_descriptor("shape");
    }

    #[test]
    #[should_panic(expected = "is assigned in more than one place")]
    fn test_load_tensor_descriptor_dyn_offsets_rejected() {
        compile_dyn_descriptor("load");
    }

    #[test]
    #[should_panic(expected = "is assigned in more than one place")]
    fn test_store_tensor_descriptor_dyn_offsets_rejected() {
        compile_dyn_descriptor("store");
    }

    // teenyc-2m7: a local with no SSA form that is assigned in more than one place must fail to
    // compile instead of silently using one assignment's value. A codegen error panics inside
    // rustc and reaches the test through `run_compiler`.
    fn compile_multi_def(kind: &str) {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let file =
            env::current_dir().unwrap().join(format!("tests/data/triton_multi_def_{kind}.rs"));
        let _ = compiler.compile(&file, "nvptx64-nvidia-cuda", &format!("test_multi_def_{kind}"));
    }

    #[test]
    #[should_panic(expected = "is assigned in more than one place")]
    fn test_multi_def_const_array_rejected() {
        compile_multi_def("const_array");
    }

    #[test]
    #[should_panic(expected = "is assigned in more than one place")]
    fn test_multi_def_dyn_array_rejected() {
        compile_multi_def("dyn_array");
    }

    #[test]
    #[should_panic(expected = "is assigned in more than one place")]
    fn test_multi_def_block_ptr_rejected() {
        compile_multi_def("block_ptr");
    }

    #[test]
    #[should_panic(expected = "is assigned in more than one place")]
    fn test_multi_def_option_rejected() {
        compile_multi_def("option");
    }

    #[test]
    #[should_panic(expected = "is assigned in more than one place")]
    fn test_multi_def_tuple_rejected() {
        compile_multi_def("tuple");
    }
}
