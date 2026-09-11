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
}
