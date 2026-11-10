use std::env;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

use cargo_metadata::MetadataCommand;
use rustc_helper::llvm::Llvm;
use rustc_helper::triton::Triton;

/// Cross-builds SLEEF (`src/sleef`) into a static riscv64 `libsleef.a`, which RiscvBackend
/// links into kernels so their vector math calls SLEEF's RVV functions rather than libm.
///
/// SLEEF generates headers with tools it builds for the host, so it is configured twice: a
/// native build for those tools, then the riscv64 one pointing at it. The cross build uses
/// clang (`$TEENY_SLEEF_CC`/`$TEENY_SLEEF_CXX` to override), since gcc only gained the RVV
/// intrinsics SLEEF needs in version 14, and the riscv64 libc headers (Ubuntu's
/// `libc6-dev-riscv64-cross`). Only the archive is built, so no riscv64 C++ library is needed.
///
/// Returns `None`, after a cargo warning, when it can't be built: RiscvBackend then keeps
/// lowering math to libm, so a host without the cross toolchain still gets a working compiler.
fn build_sleef_riscv(root_dir: &Path) -> Option<PathBuf> {
    let source = root_dir.join("src/sleef");
    let build_dir = root_dir.join("target/build/sleef-build");
    println!("cargo:rerun-if-env-changed=TEENY_SLEEF_CC");
    println!("cargo:rerun-if-env-changed=TEENY_SLEEF_CXX");
    // SLEEF's cross toolchain file only accepts a compiler given by full path.
    let find_tool = |var: &str, default: &str| -> Option<PathBuf> {
        let tool = PathBuf::from(env::var(var).unwrap_or_else(|_| default.to_string()));
        if tool.is_absolute() {
            return Some(tool);
        }
        env::split_paths(&env::var_os("PATH")?).map(|dir| dir.join(&tool)).find(|p| p.is_file())
    };
    let (Some(cc), Some(cxx)) =
        (find_tool("TEENY_SLEEF_CC", "clang"), find_tool("TEENY_SLEEF_CXX", "clang++"))
    else {
        println!(
            "cargo:warning=clang/clang++ not found on PATH (set $TEENY_SLEEF_CC/$TEENY_SLEEF_CXX); \
             RISC-V kernels will call libm for vector math."
        );
        return None;
    };
    let (cc, cxx) = (cc.display(), cxx.display());

    if !source.join("CMakeLists.txt").exists() {
        println!(
            "cargo:warning=SLEEF source missing at {}: run `git submodule update --init \
             src/sleef`. RISC-V kernels will call libm for vector math.",
            source.display()
        );
        return None;
    }

    let common = [
        "-GNinja",
        "-DCMAKE_BUILD_TYPE=Release",
        "-DBUILD_SHARED_LIBS=OFF",
        "-DSLEEF_BUILD_TESTS=OFF",
        "-DSLEEF_BUILD_DFT=OFF",
        "-DSLEEF_BUILD_QUAD=OFF",
        "-DSLEEF_BUILD_GNUABI_LIBS=OFF",
        "-DSLEEF_BUILD_SCALAR_LIB=OFF",
        "-DSLEEF_ENABLE_TLFLOAT=OFF",
        "-DSLEEF_ENABLE_TESTER4=OFF",
        "-DSLEEF_SHOW_CONFIG=OFF",
    ];
    let native_dir = build_dir.join("native");
    let riscv_dir = build_dir.join("riscv64");
    let steps: [(&str, Vec<String>); 4] = [
        (
            "configure (native)",
            [
                vec![
                    "-S".into(),
                    source.display().to_string(),
                    "-B".into(),
                    native_dir.display().to_string(),
                    format!("-DCMAKE_C_COMPILER={cc}"),
                    format!("-DCMAKE_CXX_COMPILER={cxx}"),
                ],
                common.iter().map(|s| s.to_string()).collect(),
            ]
            .concat(),
        ),
        ("build (native)", vec!["--build".into(), native_dir.display().to_string()]),
        (
            "configure (riscv64)",
            [
                vec![
                    "-S".into(),
                    source.display().to_string(),
                    "-B".into(),
                    riscv_dir.display().to_string(),
                    format!(
                        "-DCMAKE_TOOLCHAIN_FILE={}",
                        source.join("toolchains/riscv64-llvm.cmake").display()
                    ),
                    format!("-DCMAKE_C_COMPILER={cc}"),
                    format!("-DCMAKE_CXX_COMPILER={cxx}"),
                    "-DCMAKE_CXX_COMPILER_TARGET=riscv64-linux-gnu".into(),
                    // Checks compile without linking: there is no riscv64 libstdc++ to link.
                    "-DCMAKE_TRY_COMPILE_TARGET_TYPE=STATIC_LIBRARY".into(),
                    // Kernels are shared libraries.
                    "-DCMAKE_POSITION_INDEPENDENT_CODE=ON".into(),
                    format!("-DNATIVE_BUILD_DIR={}", native_dir.display()),
                    "-DSLEEF_ENFORCE_RVVM1=ON".into(),
                    "-DSLEEF_ENFORCE_RVVM2=ON".into(),
                ],
                common.iter().map(|s| s.to_string()).collect(),
            ]
            .concat(),
        ),
        (
            "build (riscv64)",
            vec![
                "--build".into(),
                riscv_dir.display().to_string(),
                "--target".into(),
                "sleef".into(),
            ],
        ),
    ];

    for (step, args) in &steps {
        let output = match Command::new("cmake").args(args).env("DESTDIR", "").output() {
            Ok(output) => output,
            Err(e) => {
                println!("cargo:warning=SLEEF {step} could not run cmake: {e}");
                return None;
            }
        };
        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let stdout = String::from_utf8_lossy(&output.stdout);
            let tail: Vec<&str> = stdout.lines().chain(stderr.lines()).rev().take(10).collect();
            println!(
                "cargo:warning=SLEEF {step} failed ({}); RISC-V kernels will call libm for \
                 vector math. Last output:",
                output.status
            );
            for line in tail.iter().rev() {
                println!("cargo:warning=  {line}");
            }
            return None;
        }
    }

    let lib = riscv_dir.join("lib/libsleef.a");
    if !lib.exists() {
        println!("cargo:warning=SLEEF built, but {} is missing", lib.display());
        return None;
    }
    println!("cargo:rerun-if-changed={}", source.join("src").display());
    Some(lib)
}

fn main() -> Result<(), Box<dyn Error>> {
    let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());

    // Get project directory and target directory using cargo_metadata
    // Use rustc_llvm directory since that's where llvm.toml and triton.toml are located
    let metadata = MetadataCommand::new().exec().unwrap();
    let root_dir: PathBuf = metadata.workspace_root.into();
    let target_dir: PathBuf = metadata.target_directory.into();

    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").unwrap());
    let wrapper_dir = manifest_dir.join("mlir-wrapper");

    // Get LLVM and Triton configs using helper functions
    let llvm = Llvm::new(&root_dir, &target_dir);
    let triton = Triton::new(&root_dir, &target_dir);

    // Use LLVM install_dir (the output of `llvm-config --prefix`) for cmake config paths --
    // it already directly contains `lib/cmake/{llvm,mlir}`, `lib/`, and `include/`.
    let triton_build_dir = triton.out_dir.join("build");
    let mlir_dir = llvm.install_dir.join("lib/cmake/mlir");

    let mlir_wrapper_build_dir = root_dir.join("target/build/mlir-wrapper-build");

    // Ensure the mlir-wrapper build directory exists before configuring cmake
    if let Err(e) = std::fs::create_dir_all(&mlir_wrapper_build_dir) {
        panic!("Failed to create mlir-wrapper build directory {:?}: {}", mlir_wrapper_build_dir, e);
    }

    let sleef_riscv_lib = build_sleef_riscv(&root_dir);

    // Configure cmake build
    let mut config = cmake::Config::new(&wrapper_dir);

    config
        .generator("Ninja")
        .out_dir(mlir_wrapper_build_dir)
        // Don't let a `DESTDIR` inherited from the environment redirect `cmake --install`
        // out from under the absolute `CMAKE_INSTALL_PREFIX` we set via `out_dir` above --
        // see the same guard in bootstrap's LLVM build step (build_steps/llvm.rs).
        .env("DESTDIR", "")
        .define("CMAKE_BUILD_TYPE", "Release")
        .define("LLVM_DIR", llvm.install_dir.join("lib/cmake/llvm"))
        .define("MLIR_DIR", &mlir_dir)
        .define("TRITON_SOURCE_DIR", triton.source_dir())
        .define("TRITON_BUILD_DIR", triton_build_dir)
        // The CMake option defaults OFF for standalone configures; passing it
        // explicitly also overrides a stale OFF left in an existing build cache.
        .define("TRITON_CPU_ENABLE", "ON")
        // Always defined, so a build without SLEEF overrides a path cached by an earlier one.
        .define(
            "TEENY_SLEEF_RISCV_LIB",
            sleef_riscv_lib.as_ref().map(|lib| lib.display().to_string()).unwrap_or_default(),
        );

    let dst = config.build();

    println!(
        "cargo:rustc-env=LLVM_INCLUDE_DIRECTORY={}",
        llvm.install_dir.join("include").display()
    );

    println!(
        "cargo:rustc-env=TRITON_INCLUDE_DIRECTORY={}",
        triton.source_dir().join("include").display()
    );

    // Link the built library
    println!("cargo:rustc-link-search=native={}", dst.join("lib").display());
    println!("cargo:rustc-link-lib=static=mlir-wrapper");
    // The wrapper's RISC-V backend references the in-tree RVV dialect, which cmake
    // builds as its own archive. It must follow mlir-wrapper on the link line.
    println!("cargo:rustc-link-lib=static=MLIRRVVDialect");
    // RiscvBackend lowers through TritonCPU (mlir-wrapper's CpuBackend), whose
    // passes live in these archives. Dependents come first so each archive's
    // references resolve against the ones after it; the MLIR libraries they
    // need are linked by mlir-sys, and TritonIR by `triton` below.
    for lib in ["TritonCPUToLLVM", "TritonCPUTransforms", "TritonToTritonCPU", "TritonCPUIR"] {
        println!("cargo:rustc-link-lib=static={lib}");
    }

    // Link MLIR libraries
    let mlir_lib_dir = llvm.install_dir.join("lib");
    println!("cargo:rustc-link-search=native={}", mlir_lib_dir.display());

    // Link LLVM support libraries
    println!("cargo:rustc-link-lib=LLVMSupport");
    println!("cargo:rustc-link-lib=LLVMCore");

    // Link Triton libraries
    println!("cargo:rustc-link-search=native={}", triton.link_dir().display());
    for lib in triton.link_libs() {
        println!("cargo:rustc-link-lib={}", lib);
    }

    // Link C++ standard library
    if cfg!(target_os = "macos") {
        println!("cargo:rustc-link-lib=c++");
    } else {
        println!("cargo:rustc-link-lib=stdc++");
    }

    // Rerun if wrapper sources change
    println!("cargo:rerun-if-changed=mlir-wrapper/");
    // Note: llvm.toml and triton.toml are in rustc_llvm directory, tracked by rustc_llvm's build.rs

    // Generate bindings header path for use in lib.rs
    println!("cargo:include={}", wrapper_dir.display());
    println!("cargo:out_dir={}", out_dir.display());

    Ok(())
}
