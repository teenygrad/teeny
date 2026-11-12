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
    fn test_triton_relu() -> Result<(), Box<dyn std::error::Error>> {
        // Set RUST_LOG=debug in environment to see debug output
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        let compiler = LlvmCompiler::new();
        let tensor_add = env::current_dir().unwrap().join("tests/data/triton_relu.rs");
        let target = "nvptx64-nvidia-cuda";

        println!("Compiling tensor add with target: {}", tensor_add.display());
        compiler.compile(&tensor_add, target, "test_triton_relu")?;

        Ok(())
    }

    // teenyc-2ef (teenygrad#133): every kernel argument used to be declared `tt.divisibility =
    // 16`. For a scalar like `n_elements` that's false for most launches, and Triton then treats
    // each thread's 4 contiguous lanes as one for anything derived from it: the bounds mask (and
    // any division by a scalar) is computed once per group of 4, so masked stores run past the
    // end. Pointer arguments keep the hint; scalars must not carry it.
    #[test]
    fn test_scalar_args_have_no_divisibility_hint() -> Result<(), Box<dyn std::error::Error>> {
        let _ = fmt()
            .with_env_filter(
                EnvFilter::try_from_default_env().unwrap_or_else(|_| EnvFilter::new("info")),
            )
            .try_init();

        // The pre-Triton MLIR is only written next to an explicitly requested object file, as
        // teenygrad's compiler driver requests it, so this compiles with `--emit=obj=...` instead
        // of `LlvmCompiler::compile`.
        let file = env::current_dir().unwrap().join("tests/data/triton_relu.rs");
        let obj = PathBuf::from("/tmp/kernel-test_relu_divisibility.o");
        let args: Vec<String> = [
            "rustc",
            &file.display().to_string(),
            "-Copt-level=3",
            "-Cpanic=abort",
            &format!("--emit=obj={}", obj.display()),
            "--target=nvptx64-nvidia-cuda",
            "--crate-type=lib",
            "-Coverflow-checks=off",
            "--frontend=triton",
        ]
        .map(String::from)
        .to_vec();
        run_compiler(&args, &mut MlirBackendCallbacks);
        let mlir = std::fs::read_to_string(obj.with_extension("mlir"))?;

        let signature = mlir
            .lines()
            .find(|l| l.contains("@entry_point"))
            .ok_or_else(|| format!("no entry_point in MLIR:\n{mlir}"))?;
        // `%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: ..., %arg2: i32 ...`
        let args: Vec<&str> = signature.split("%arg").skip(1).collect();
        assert_eq!(args.len(), 3, "expected 3 entry_point args in:\n{signature}");
        for arg in &args[..2] {
            assert!(
                arg.contains("!tt.ptr<f32> {tt.divisibility = 16 : i32}"),
                "pointer arg lost its divisibility hint:\n{signature}"
            );
        }
        assert!(
            args[2].starts_with("2: i32") && !args[2].contains("tt.divisibility"),
            "scalar n_elements must not carry a divisibility hint:\n{signature}"
        );
        Ok(())
    }
}
