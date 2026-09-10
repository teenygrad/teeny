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

//! MLIR codegen backend implementation.
//!
//! This module provides the main backend implementation that integrates
//! with rustc's compilation pipeline.

use std::any::Any;
use std::ffi::CString;
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use melior::pass;
use melior::pass::PassManager;
use rustc_codegen_ssa::back::lto::{ThinModule, ThinShared};
use rustc_codegen_ssa::back::write::{
    CodegenContext, FatLtoInput, ModuleConfig, SharedEmitter, TargetMachineFactoryFn, ThinLtoInput,
};
use rustc_codegen_ssa::base::codegen_crate;
use rustc_codegen_ssa::traits::*;
use rustc_codegen_ssa::{CompiledModule, CompiledModules, CrateInfo, ModuleCodegen, TargetConfig};
use rustc_data_structures::fx::FxIndexMap;
use rustc_data_structures::profiling::SelfProfilerRef;
use rustc_errors::DiagCtxtHandle;
use rustc_middle::dep_graph;
use rustc_middle::dep_graph::{WorkProduct, WorkProductId};
use rustc_middle::ty::TyCtxt;
use rustc_session::Session;
use rustc_session::config::{self, OutputFilenames, PrintKind, PrintRequest};
use rustc_span::Symbol;
use rustc_target::spec::Arch;
use tracing::info;

use crate::mlir::MlirModule;
use crate::mlir::codegen::Codegen;
use crate::mlir::codegen::triton::TritonCodegen;
use crate::mlir::errors::MlirError;

/// The MLIR codegen backend.
#[derive(Copy, Clone)]
pub struct MlirCodegenBackend {}

impl MlirCodegenBackend {
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> Box<dyn CodegenBackend> {
        Box::new(MlirCodegenBackend {})
    }
}

impl ExtraBackendMethods for MlirCodegenBackend {
    fn codegen_allocator<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        module_name: &str,
        _methods: &[rustc_ast::expand::allocator::AllocatorMethod],
    ) -> Self::Module {
        info!("=== MLIR codegen_allocator ===");
        info!("Module name: {}", module_name);

        // Create a placeholder module for the allocator
        MlirModule::new(module_name)
    }

    fn compile_codegen_unit(
        &self,
        tcx: TyCtxt<'_>,
        cgu_name: Symbol,
    ) -> (ModuleCodegen<Self::Module>, u64) {
        let start_time = Instant::now();

        let dep_node = tcx.codegen_unit(cgu_name).codegen_dep_node(tcx);
        let (module, _) = tcx.dep_graph.with_task(
            dep_node,
            tcx,
            || compile_codegen_unit_impl(tcx, cgu_name),
            Some(dep_graph::hash_result),
        );

        let time_to_codegen = start_time.elapsed();
        let cost = time_to_codegen.as_nanos() as u64;

        (module, cost)
    }
}

fn compile_codegen_unit_impl(
    tcx: TyCtxt<'_>,
    cgu_name: Symbol,
) -> ModuleCodegen<MlirModule<'static>> {
    let cgu = tcx.codegen_unit(cgu_name);

    info!("========================================");
    info!("=== MLIR compile_codegen_unit ===");
    info!("CGU name: {}", cgu_name);
    info!("CGU size estimate: {}", cgu.size_estimate());
    info!("========================================");

    // Create the MLIR module: backend (Cuda/Riscv) and compile options are
    // resolved from the session's --target architecture and -C target-cpu
    // (see crate::mlir::target::resolve).
    let mut mlir_module = MlirModule::new_for_session(cgu_name.as_str(), tcx.sess);
    let mut triton_codegen = TritonCodegen::new(&mlir_module);

    // Get all mono items in deterministic order
    let mono_items = cgu.items_in_deterministic_order(tcx);

    info!("--- Mono Items ({}) ---", mono_items.len());

    // Create a MIR visitor for detailed logging
    // let mut visitor = MirVisitor::new(tcx);
    // eprintln!("[DEBUG] Created MirVisitor");

    for (idx, (mono_item, data)) in mono_items.iter().enumerate() {
        info!("");
        info!("=== Mono Item [{}/{}] ===", idx + 1, mono_items.len());
        info!("Linkage: {:?}", data.linkage);
        info!("Visibility: {:?}", data.visibility);

        triton_codegen.codegen(tcx, mono_item).expect("Failed to generate MLIR for instance");
    }

    cleanup_mlir_module(&mut mlir_module).expect("MLIR cleanup passes failed");

    mlir_module.mlir_source = Some(mlir_module.llmod().as_operation().to_string());

    compile_module(&mut mlir_module).expect("Triton passes failed");

    info!("");
    info!("========================================");
    info!("=== End of CGU: {} ===", cgu_name);
    info!("========================================");

    ModuleCodegen::new_regular(cgu_name.to_string(), mlir_module)
}

fn cleanup_mlir_module(mlir_module: &mut MlirModule<'static>) -> Result<(), MlirError> {
    let pass_manager = PassManager::new(mlir_module.context());

    pass_manager.add_pass(pass::transform::create_canonicalizer_pass());
    pass_manager.add_pass(pass::transform::create_symbol_dce_pass());

    pass_manager
        .run(mlir_module.llmod_mut())
        .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;

    Ok(())
}

fn compile_module(mlir_module: &mut MlirModule<'static>) -> Result<(), MlirError> {
    let ok = mlir_module.compiler.compile(mlir_module.llmod().to_raw());
    if !ok {
        return Err(MlirError::CodegenFailed { err: "Triton compilation failed".to_string() });
    }

    let asm = mlir_module
        .compiler
        .get_asm()
        .ok_or_else(|| MlirError::CodegenFailed { err: "Triton returned no ASM".to_string() })?
        .to_owned();

    log_pipeline_stages(mlir_module, &asm);

    let metadata = crate::mlir::module::KernelMetadata::parse(&asm);
    mlir_module.kernel_metadata = Some(metadata);
    mlir_module.asm = Some(asm);
    // Some backends (e.g. RiscvBackend) additionally link a real binary
    // (a shared library) in makeBIN; write_compiled_module prefers this
    // over asm when present, so the actual final artifact -- not just
    // its assembly text -- reaches the output file.
    mlir_module.compiled_bin = mlir_module.compiler.get_bin_bytes().map(<[u8]>::to_vec);
    Ok(())
}

/// Logs each MLIR pipeline stage's IR (mlir, ttir, ttgpuir, llir, llvmir,
/// ptx/asm) at `debug` level, once per stage. Per-pass IR within
/// ttir/ttgpuir/llir is controlled separately by `CompileOptions::debug` (see
/// `MlirModule::new_with_capability`), which the C++ backend consults at
/// `trace` level.
///
/// `mlir` is `mlir_module.mlir_source` — the generic-dialect MLIR this
/// backend's own codegen produced from the mono items (post canonicalizer /
/// symbol-dce cleanup), captured before `compiler.compile()` below converts
/// it to Triton IR. It's the earliest IR snapshot available: the state right
/// before ttir, rather than ttir itself.
///
/// A caller turns this on via `RUSTC_LOG=rustc_codegen_llvm::mlir=debug` (or
/// `=trace` for the per-pass detail too). See
/// [`crate::mlir::module::MlirModule::new_for_session`] for how
/// `CompileOptions::debug` gets set from this same log target.
fn log_pipeline_stages(mlir_module: &MlirModule<'static>, asm: &str) {
    let compiler = &mlir_module.compiler;

    tracing::debug!(target: crate::mlir::LOG_TARGET, stage = "mlir", "{}", mlir_module.mlir_source.as_deref().unwrap_or_default());
    tracing::debug!(target: crate::mlir::LOG_TARGET, stage = "ttir", "{}", compiler.get_ttir().unwrap_or_default());
    tracing::debug!(target: crate::mlir::LOG_TARGET, stage = "ttgir", "{}", compiler.get_ttgir().unwrap_or_default());
    tracing::debug!(target: crate::mlir::LOG_TARGET, stage = "llir", "{}", compiler.get_llir().unwrap_or_default());
    tracing::debug!(target: crate::mlir::LOG_TARGET, stage = "llvmir", "{}", compiler.get_llvm_ir().unwrap_or_default());
    tracing::debug!(target: crate::mlir::LOG_TARGET, stage = "asm", "{}", asm);
}

impl WriteBackendMethods for MlirCodegenBackend {
    type Module = MlirModule<'static>;
    type ModuleBuffer = ModuleBuffer;
    type TargetMachine = ();
    type ThinData = ThinData;

    fn target_machine_factory(
        &self,
        _sess: &Session,
        _opt_level: config::OptLevel,
        _target_features: &[String],
    ) -> TargetMachineFactoryFn<Self> {
        // The MLIR backend has no LLVM TargetMachine of its own (Self::TargetMachine = ());
        // target-specific configuration is threaded through MlirModule instead.
        Arc::new(move |_dcx, _config| ())
    }

    /// Performs fat LTO by merging all modules into a single one, running autodiff
    /// if necessary and running any further optimizations
    fn optimize_and_codegen_fat_lto(
        _sess: &Session,
        _cgcx: &CodegenContext,
        _shared_emitter: &SharedEmitter,
        _tm_factory: TargetMachineFactoryFn<Self>,
        _exported_symbols_for_lto: &[String],
        _each_linked_rlib_for_lto: &[PathBuf],
        _modules: Vec<FatLtoInput<Self>>,
    ) -> CompiledModule {
        todo!("Not implemented");
    }

    /// Performs thin LTO by performing necessary global analysis and returning two
    /// lists, one of the modules that need optimization and another for modules that
    /// can simply be copied over from the incr. comp. cache.
    ///
    /// The MLIR/Triton backend compiles each codegen unit to final PTX
    /// independently inside `compile_codegen_unit_impl`, so there's no
    /// cross-module bitcode merging to do here: this is a pass-through that
    /// wraps each "red" (needs-work) module unchanged and forwards each
    /// "green" (cached) module's `WorkProduct` directly.
    fn run_thin_lto(
        _cgcx: &CodegenContext,
        _prof: &SelfProfilerRef,
        _dcx: DiagCtxtHandle<'_>,
        _exported_symbols_for_lto: &[String],
        _each_linked_rlib_for_lto: &[PathBuf],
        modules: Vec<ThinLtoInput<Self>>,
    ) -> (Vec<ThinModule<Self>>, Vec<WorkProduct>) {
        let mut work_products = Vec::new();
        let mut serialized_modules = Vec::new();
        let mut module_names = Vec::new();

        for input in modules {
            match input {
                ThinLtoInput::Red { name, buffer } => {
                    module_names.push(CString::new(name).unwrap());
                    serialized_modules.push(buffer);
                }
                ThinLtoInput::Green { wp, .. } => {
                    work_products.push(wp);
                }
            }
        }

        let num_modules = module_names.len();
        let shared =
            Arc::new(ThinShared { data: ThinData {}, modules: serialized_modules, module_names });
        let thin_modules =
            (0..num_modules).map(|idx| ThinModule { shared: Arc::clone(&shared), idx }).collect();

        (thin_modules, work_products)
    }

    fn optimize(
        _cgcx: &CodegenContext,
        _prof: &SelfProfilerRef,
        _shared_emitter: &SharedEmitter,
        module: &mut ModuleCodegen<Self::Module>,
        _config: &ModuleConfig,
    ) {
        info!("MLIR: optimize module '{}'", module.name);
        // Triton's own optimization pipeline already ran inside
        // compile_codegen_unit_impl (via cleanup_mlir_module/compile_module)
        // before this hook is called, so there's nothing further to do here.
    }

    fn optimize_and_codegen_thin(
        cgcx: &CodegenContext,
        _prof: &SelfProfilerRef,
        _shared_emitter: &SharedEmitter,
        _tm_factory: TargetMachineFactoryFn<Self>,
        thin: ThinModule<Self>,
    ) -> CompiledModule {
        let name = thin.name().to_string();
        info!("=== MLIR optimize_and_codegen_thin '{}' (pass-through) ===", name);

        // Recover the bytes serialize_module serialized into the thin buffer
        // (compiled_bin if the backend produced one, else the asm text --
        // see serialize_module). write_compiled_module treats compiled_bin
        // opaquely, so this round-trip is correct either way.
        let mut mlir_module = MlirModule::new(&name);
        mlir_module.compiled_bin = Some(thin.data().to_vec());

        write_compiled_module(cgcx, ModuleCodegen::new_regular(name, mlir_module))
    }

    fn codegen(
        cgcx: &CodegenContext,
        _prof: &SelfProfilerRef,
        _shared_emitter: &SharedEmitter,
        module: ModuleCodegen<Self::Module>,
        _config: &ModuleConfig,
    ) -> CompiledModule {
        info!("=== MLIR codegen ===");
        write_compiled_module(cgcx, module)
    }

    fn serialize_module(module: Self::Module, _is_thin: bool) -> Self::ModuleBuffer {
        // Prefer compiled_bin (e.g. RiscvBackend's linked shared library)
        // over the asm text, matching write_compiled_module's
        // precedence -- so optimize_and_codegen_thin's pass-through
        // preserves the backend's actual final artifact, not just its
        // assembly text.
        let bytes = module
            .compiled_bin
            .unwrap_or_else(|| module.asm.map(|s| s.into_bytes()).unwrap_or_default());
        ModuleBuffer { data: bytes }
    }
}

/// Writes a compiled module's final artifact (and, if captured, its
/// pre-Triton MLIR source) to the expected output paths and builds the
/// resulting `CompiledModule`. Shared by `codegen` (no-LTO path) and
/// `optimize_and_codegen_thin` (ThinLTO pass-through path) since both end up
/// with a `ModuleCodegen<MlirModule>` whose `asm`/`compiled_bin` are
/// already populated by compile_codegen_unit_impl (via compile_module).
///
/// `compiled_bin` (e.g. RiscvBackend's linked ELF shared library) is
/// preferred over `asm` when present, so the backend's actual final
/// artifact reaches the output file rather than just its assembly text.
/// `serialize_module`/`optimize_and_codegen_thin` round-trip whichever of
/// the two was preferred through the ThinLTO pass-through path too.
fn write_compiled_module(
    cgcx: &CodegenContext,
    module: ModuleCodegen<MlirModule<'static>>,
) -> CompiledModule {
    info!("Module name: {}", module.name);

    let bytes: &[u8] = if let Some(bin) = module.module_llvm.compiled_bin.as_deref() {
        bin
    } else {
        module
            .module_llvm
            .asm
            .as_deref()
            .unwrap_or_else(|| {
                panic!(
                    "No output available for module '{}' — Triton compilation may not have run",
                    module.name
                )
            })
            .as_bytes()
    };

    let out_path = cgcx
        .output_filenames
        .temp_path_for_cgu(rustc_session::config::OutputType::Object, &module.name);
    std::fs::write(&out_path, bytes)
        .unwrap_or_else(|e| panic!("Failed to write output to {}: {}", out_path.display(), e));
    info!("Output written to {} ({} bytes)", out_path.display(), bytes.len());

    if let Some(mlir_src) = module.module_llvm.mlir_source.as_deref() {
        let mlir_path = cgcx
            .output_filenames
            .path(rustc_session::config::OutputType::Object)
            .as_path()
            .with_extension("mlir");
        std::fs::write(&mlir_path, mlir_src.as_bytes())
            .unwrap_or_else(|e| panic!("Failed to write MLIR to {}: {}", mlir_path.display(), e));
        info!("MLIR written to {} ({} bytes)", mlir_path.display(), mlir_src.len());
    }

    CompiledModule {
        name: module.name,
        kind: module.kind,
        object: Some(out_path),
        dwarf_object: None,
        bytecode: None,
        assembly: None,
        llvm_ir: None,
        links_from_incr_cache: Vec::new(),
    }
}

impl CodegenBackend for MlirCodegenBackend {
    fn name(&self) -> &'static str {
        "mlir"
    }

    fn target_cpu(&self, sess: &Session) -> String {
        crate::llvm_util::target_cpu(sess).to_string()
    }

    fn target_config(&self, _sess: &Session) -> TargetConfig {
        // To Do: Implement MLIR-specific target config for the target
        // defined in the session
        TargetConfig {
            target_features: Vec::new(),
            unstable_target_features: Vec::new(),
            has_reliable_f16: false,
            has_reliable_f16_math: false,
            has_reliable_f128: false,
            has_reliable_f128_math: false,
        }
    }

    fn codegen_crate<'tcx>(&self, tcx: TyCtxt<'tcx>) -> Box<dyn Any> {
        info!("========================================");
        info!("=== MLIR codegen_crate ===");
        info!("Crate name: {:?}", tcx.crate_name(rustc_hir::def_id::LOCAL_CRATE));
        info!("========================================");

        Box::new(codegen_crate(self.clone(), tcx))
    }

    fn join_codegen(
        &self,
        ongoing_codegen: Box<dyn Any>,
        sess: &Session,
        _outputs: &OutputFilenames,
        crate_info: &CrateInfo,
    ) -> (CompiledModules, FxIndexMap<WorkProductId, WorkProduct>) {
        info!("=== MLIR join_codegen ===");

        let (compiled_modules, work_products) = ongoing_codegen
            .downcast::<rustc_codegen_ssa::back::write::OngoingCodegen<MlirCodegenBackend>>()
            .expect("Expected OngoingCodegen<MlirCodegenBackend>")
            .join(sess, crate_info);

        info!("Codegen completed");
        info!("  Work products: {}", work_products.len());

        (compiled_modules, work_products)
    }

    fn link(
        &self,
        sess: &Session,
        compiled_modules: CompiledModules,
        _crate_info: CrateInfo,
        _metadata: rustc_metadata::EncodedMetadata,
        outputs: &OutputFilenames,
    ) {
        use rustc_session::config::OutputType;
        info!("MLIR: link (writing backend output)");

        // produce_final_output_artifacts only copies temp files for output
        // types listed in --emit. There is no linking step here, so we copy
        // each module's object (asm text for Cuda, or a linked binary for
        // backends like Riscv -- see MlirModule::asm/compiled_bin) directly
        // to the -o destination.
        let out = outputs.path(OutputType::Object);
        for module in &compiled_modules.modules {
            if let Some(obj) = &module.object {
                if let Err(e) = std::fs::copy(obj, out.as_path()) {
                    sess.dcx().fatal(format!(
                        "failed to write output to {}: {}",
                        out.as_path().display(),
                        e
                    ));
                }
                info!("Output written to {}", out.as_path().display());
            }
        }
    }

    fn print(&self, req: &PrintRequest, out: &mut String, sess: &Session) {
        match req.kind {
            PrintKind::TargetCPUs => {
                out.push_str("MLIR backend target CPUs:\n");
                match &sess.target.arch {
                    Arch::Nvptx64 => {
                        out.push_str(
                            "  an NVIDIA SM version in the form `sm_<NN>` or `sm_<NN>a`, \
                             e.g. sm_70, sm_80, sm_90, sm_90a, sm_100a, sm_120a\n",
                        );
                    }
                    Arch::RiscV32 | Arch::RiscV64 => {
                        out.push_str(
                            "  a RISC-V chip identifier understood by this backend's (currently \
                             stub) RISC-V/Triton path, e.g. generic-rvv1.0, spacemit-k3 -- not \
                             an LLVM -mcpu value\n",
                        );
                    }
                    other => {
                        out.push_str(&format!(
                            "  (unsupported target architecture `{}` for the `mlir` backend)\n",
                            other.desc()
                        ));
                    }
                }
            }
            PrintKind::TargetFeatures => {
                out.push_str("MLIR backend target features:\n");
                out.push_str("  (not yet modeled separately from target-cpu)\n");
            }
            _ => {
                // Delegate other print requests to LLVM
            }
        }
    }
}

// Placeholder type for ModuleBuffer

pub struct ModuleBuffer {
    data: Vec<u8>,
}

impl ModuleBuffer {
    pub fn new() -> Self {
        Self { data: Vec::new() }
    }
}

impl rustc_codegen_ssa::traits::ModuleBufferMethods for ModuleBuffer {
    fn data(&self) -> &[u8] {
        &self.data
    }
}

pub struct ThinData {
    // TODO: Add actual thin data fields
}

unsafe impl Send for ThinData {}
unsafe impl Sync for ThinData {}

// Export the backend entry point
#[unsafe(no_mangle)]
pub fn __rustc_codegen_backend() -> Box<dyn CodegenBackend> {
    MlirCodegenBackend::new()
}
