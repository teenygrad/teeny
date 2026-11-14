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

use std::collections::HashMap;

use melior::ir::attribute::{DenseElementsAttribute, FloatAttribute, IntegerAttribute};
use melior::ir::operation::OperationLike;
use melior::ir::r#type::{IntegerType, RankedTensorType, ShapedTypeLike, TypeLike};
use melior::ir::{BlockLike, BlockRef, Location, Operation, Value, ValueLike};
use rustc_middle::mir::{
    BasicBlock, Body, CallSource, Local, Operand, Place, SwitchTargets, Terminator, TerminatorKind,
    UnwindAction,
};
use rustc_middle::ty::print::with_no_trimmed_paths;
use rustc_middle::ty::{self, EarlyBinder, Instance, TyCtxt, TyKind, TypingEnv};
use rustc_mlir::shared::arith::{Predicate, create_cmpi};
use rustc_mlir::shared::cf::create_cf_br;
use rustc_mlir::triton::call;
use rustc_span::Span;
use rustc_span::Spanned;

use crate::mlir::codegen::triton::location::span_to_location;
use crate::mlir::codegen::triton::{CodegenState, OptionValue, PhiArg, TritonCodegen};
use crate::mlir::errors::MlirError;

// Used inside codegen_terminator_call where 'a and 'tcx are concrete — no HRTB needed.
type LocalCallHandler<'a, 'tcx> = fn(
    &TritonCodegen<'a>,
    TyCtxt<'tcx>,
    &Instance<'tcx>,
    &Body<'tcx>,
    &Operand<'tcx>,
    &str,
    &[Spanned<Operand<'tcx>>],
    &Place<'tcx>,
    &Option<BasicBlock>,
    &UnwindAction,
    &CallSource,
    &Span,
    Location<'a>,
    &BlockRef<'a, 'a>,
    &mut CodegenState<'a, 'a>,
) -> Result<Option<Value<'a, 'a>>, MlirError>;

impl<'a> TritonCodegen<'a> {
    pub fn codegen_terminator<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        terminator: &Terminator<'tcx>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
        basic_blocks: &HashMap<BasicBlock, BlockRef<'a, 'a>>,
    ) -> Result<(), MlirError> {
        let location = span_to_location(self.module.context(), tcx, terminator.source_info.span);

        match &terminator.kind {
            rustc_middle::mir::TerminatorKind::Return => {
                self.codegen_return(location, terminator, mlir_block, state)
            }
            rustc_middle::mir::TerminatorKind::Goto { target } => {
                self.codegen_goto(location, target, mlir_block, basic_blocks, state)
            }
            rustc_middle::mir::TerminatorKind::SwitchInt { discr, targets } => self
                .codegen_switch_int(
                    tcx,
                    instance,
                    mir,
                    discr,
                    targets,
                    location,
                    mlir_block,
                    basic_blocks,
                    state,
                ),
            rustc_middle::mir::TerminatorKind::UnwindResume => todo!("UnwindResume"),
            rustc_middle::mir::TerminatorKind::UnwindTerminate(unwind_terminate_reason) => {
                todo!("UnwindTerminate: {:?}", unwind_terminate_reason)
            }
            rustc_middle::mir::TerminatorKind::Unreachable => todo!("Unreachable"),
            rustc_middle::mir::TerminatorKind::Drop { target, .. } => {
                // All kernel types are Copy with no destructors; treat as goto target.
                self.codegen_goto(location, target, mlir_block, basic_blocks, state)
            }
            rustc_middle::mir::TerminatorKind::Call {
                func,
                args,
                destination,
                target,
                unwind,
                call_source,
                fn_span,
            } => {
                // Use the call-site span for a more precise location on Call terminators.
                let call_loc = span_to_location(self.module.context(), tcx, *fn_span);
                self.codegen_terminator_call(
                    tcx,
                    instance,
                    mir,
                    func,
                    args,
                    destination,
                    target,
                    unwind,
                    call_source,
                    fn_span,
                    call_loc,
                    mlir_block,
                    basic_blocks,
                    state,
                )
            }
            rustc_middle::mir::TerminatorKind::TailCall { func, args, fn_span } => {
                todo!("TailCall: {:?} {:?} {:?}", func, args, fn_span)
            }
            rustc_middle::mir::TerminatorKind::Assert { target, .. } => {
                // GPU kernels have no panic infrastructure — treat Assert as an unconditional
                // branch to the success target (the assertion is assumed to hold).
                self.codegen_goto(location, target, mlir_block, basic_blocks, state)
            }
            rustc_middle::mir::TerminatorKind::Yield { .. } => todo!("Yield"),
            rustc_middle::mir::TerminatorKind::CoroutineDrop => todo!("CoroutineDrop"),
            rustc_middle::mir::TerminatorKind::FalseEdge { real_target, imaginary_target } => {
                todo!("FalseEdge: {:?} {:?}", real_target, imaginary_target)
            }
            rustc_middle::mir::TerminatorKind::FalseUnwind { real_target, unwind } => {
                todo!("FalseUnwind: {:?} {:?}", real_target, unwind)
            }
            rustc_middle::mir::TerminatorKind::InlineAsm {
                asm_macro,
                template,
                operands,
                options,
                line_spans,
                targets,
                unwind,
            } => todo!(
                "InlineAsm: {:?} {:?} {:?} {:?} {:?} {:?} {:?}",
                asm_macro,
                template,
                operands,
                options,
                line_spans,
                targets,
                unwind
            ),
        }
    }

    fn codegen_terminator_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        func: &Operand<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        target: &Option<BasicBlock>,
        unwind: &UnwindAction,
        call_source: &CallSource,
        fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        basic_blocks: &HashMap<BasicBlock, BlockRef<'a, 'a>>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        self.dispatch_call(
            tcx,
            instance,
            mir,
            func,
            args,
            destination,
            target,
            unwind,
            call_source,
            fn_span,
            location,
            mlir_block,
            state,
        )?;
        self.codegen_goto(
            location,
            &target.expect("target must be Some"),
            mlir_block,
            basic_blocks,
            state,
        )
    }

    /// Lowers a call into `mlir_block` and binds its result to `destination`, without the
    /// branch to the call's target block. Kept separate from `codegen_terminator_call` so a
    /// reduce/scan combine body, which must stay in a single block, can reuse every handler.
    pub(crate) fn dispatch_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        func: &Operand<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        target: &Option<BasicBlock>,
        unwind: &UnwindAction,
        call_source: &CallSource,
        fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        let func_name = match func {
            Operand::Constant(c) => {
                if let TyKind::FnDef(def_id, _) = c.ty().kind() {
                    with_no_trimmed_paths!(tcx.def_path_str(*def_id))
                } else {
                    format!("{:?}", func)
                }
            }
            _ => format!("XX{:?}", func),
        };

        let method: LocalCallHandler<'a, 'tcx> = match func_name.as_str() {
            "core::ops::Mul::mul" => TritonCodegen::codegen_mul_call as LocalCallHandler<'a, 'tcx>,
            "core::ops::Add::add" => TritonCodegen::codegen_add_call as LocalCallHandler<'a, 'tcx>,
            "core::ops::Sub::sub" => TritonCodegen::codegen_sub_call as LocalCallHandler<'a, 'tcx>,
            "core::ops::Neg::neg" => TritonCodegen::codegen_neg_call as LocalCallHandler<'a, 'tcx>,
            "core::ops::Div::div" => TritonCodegen::codegen_div_call as LocalCallHandler<'a, 'tcx>,
            "core::ops::BitAnd::bitand" => {
                TritonCodegen::codegen_and_call as LocalCallHandler<'a, 'tcx>
            }
            "core::ops::BitOr::bitor" => {
                TritonCodegen::codegen_or_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::program_id" => {
                TritonCodegen::codegen_program_id as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::arange" => TritonCodegen::codegen_arange as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::arange_f32" => {
                TritonCodegen::codegen_arange_f32 as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::load" => TritonCodegen::codegen_load as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::load_scalar_f32_as_i32" => {
                TritonCodegen::codegen_load_scalar_f32_as_i32 as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::store" => TritonCodegen::codegen_store as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::maximum" => {
                TritonCodegen::codegen_maximum as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::zeros" => TritonCodegen::codegen_zeros as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::zeros_like" => {
                TritonCodegen::codegen_zeros_like as LocalCallHandler<'a, 'tcx>
            }
            "transmute" | "triton_kitchen_sink::transmute" => {
                TritonCodegen::codegen_transmute_slice as LocalCallHandler<'a, 'tcx>
            }
            "triton::types::Comparison::lt" => {
                TritonCodegen::codegen_lt_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::types::Comparison::ge" => {
                TritonCodegen::codegen_ge_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::types::Comparison::gt" => {
                TritonCodegen::codegen_gt_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::types::Comparison::le" => {
                TritonCodegen::codegen_le_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::types::Comparison::eq" => {
                TritonCodegen::codegen_eq_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::types::Comparison::ne" => {
                TritonCodegen::codegen_ne_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::gt" => TritonCodegen::codegen_gt_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::ge" => TritonCodegen::codegen_ge_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::lt" => {
                TritonCodegen::codegen_triton_lt_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::le" => TritonCodegen::codegen_le_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::eq" => TritonCodegen::codegen_eq_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::ne" => TritonCodegen::codegen_ne_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::lt_scalar" => {
                TritonCodegen::codegen_lt_scalar_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::le_scalar" => {
                TritonCodegen::codegen_le_scalar_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::gt_scalar" => {
                TritonCodegen::codegen_gt_scalar_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::ge_scalar" => {
                TritonCodegen::codegen_ge_scalar_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::eq_scalar" => {
                TritonCodegen::codegen_eq_scalar_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::ne_scalar" => {
                TritonCodegen::codegen_ne_scalar_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::types::AddOffsets::add_offsets" => {
                TritonCodegen::codegen_add_ptr as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::cast" => {
                TritonCodegen::codegen_cast_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::cat" => TritonCodegen::codegen_cat_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::broadcast" => {
                TritonCodegen::codegen_broadcast_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::broadcast_to" => {
                TritonCodegen::codegen_broadcast_to_call as LocalCallHandler<'a, 'tcx>
            }
            // Same handler, same reason as `expand_dims_i32` below: the
            // handler reads its operands through `codegen_operand` and is
            // agnostic to the element type, while the Rust signatures differ
            // because `Tensor<D>` and `I32Tensor` are separate associated
            // types (teenyc-u9z).
            "triton::Triton::broadcast_to_i32" => {
                TritonCodegen::codegen_broadcast_to_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::expand_dims" => {
                TritonCodegen::codegen_expand_dims_call as LocalCallHandler<'a, 'tcx>
            }
            // Same handler: `codegen_expand_dims_call` reads `args[0]` through
            // `codegen_operand` with the operand's own type, so it is agnostic
            // to the element type. Only the Rust signature differs --
            // `expand_dims` is typed over `Self::Tensor<D>` and this one over
            // `Self::I32Tensor`, which are separate associated types with no
            // equality bound, so teenygrad needs both (teenyc-u9z).
            "triton::Triton::expand_dims_i32" => {
                TritonCodegen::codegen_expand_dims_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::permute" => {
                TritonCodegen::codegen_permute_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::reshape" => {
                TritonCodegen::codegen_reshape_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::trans" => {
                TritonCodegen::codegen_trans_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::ravel" => {
                TritonCodegen::codegen_ravel_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::view" => {
                TritonCodegen::codegen_view_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::join" => {
                TritonCodegen::codegen_join_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::interleave" => {
                TritonCodegen::codegen_interleave_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::split" => {
                TritonCodegen::codegen_split_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::dot" => TritonCodegen::codegen_dot_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::dot_scaled" => {
                TritonCodegen::codegen_dot_scaled_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::make_block_ptr" => {
                TritonCodegen::codegen_make_block_ptr_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::advance" => {
                TritonCodegen::codegen_advance_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::make_tensor_descriptor" => {
                TritonCodegen::codegen_make_tensor_descriptor_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::load_tensor_descriptor" => {
                TritonCodegen::codegen_load_tensor_descriptor_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::store_tensor_descriptor" => {
                TritonCodegen::codegen_store_tensor_descriptor_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::num_programs" => {
                TritonCodegen::codegen_num_programs_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::full" => {
                TritonCodegen::codegen_full_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::flip" => {
                TritonCodegen::codegen_flip_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::gather" => {
                TritonCodegen::codegen_gather_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::abs" => TritonCodegen::codegen_abs_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::ceil" => {
                TritonCodegen::codegen_ceil_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::floor" => {
                TritonCodegen::codegen_floor_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::cos" => TritonCodegen::codegen_cos_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::sin" => TritonCodegen::codegen_sin_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::exp" => TritonCodegen::codegen_exp_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::exp2" => {
                TritonCodegen::codegen_exp2_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::log" => TritonCodegen::codegen_log_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::log2" => {
                TritonCodegen::codegen_log2_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::rsqrt" => {
                TritonCodegen::codegen_rsqrt_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::sigmoid" => {
                TritonCodegen::codegen_sigmoid_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::sqrt" => {
                TritonCodegen::codegen_sqrt_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::sqrt_rn" => {
                TritonCodegen::codegen_sqrt_rn_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::erf" => TritonCodegen::codegen_erf_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::atan" => {
                TritonCodegen::codegen_atan_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::softmax" => {
                TritonCodegen::codegen_softmax_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::minimum" => {
                TritonCodegen::codegen_minimum_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::clamp" => {
                TritonCodegen::codegen_clamp_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::fma" => TritonCodegen::codegen_fma_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::fdiv" => {
                TritonCodegen::codegen_fdiv_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::div_rn" => {
                TritonCodegen::codegen_div_rn_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::cdiv" => {
                TritonCodegen::codegen_cdiv_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::swizzle2d" => {
                TritonCodegen::codegen_swizzle2d_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::sum" => TritonCodegen::codegen_sum_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::max" => TritonCodegen::codegen_max_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::max_with_indices" => {
                TritonCodegen::codegen_max_with_indices_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::min" => TritonCodegen::codegen_min_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::min_with_indices" => {
                TritonCodegen::codegen_min_with_indices_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::argmax" => {
                TritonCodegen::codegen_argmax_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::argmin" => {
                TritonCodegen::codegen_argmin_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::xor_sum" => {
                TritonCodegen::codegen_xor_sum_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::cumsum" => {
                TritonCodegen::codegen_cumsum_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::cumprod" => {
                TritonCodegen::codegen_cumprod_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::sort" => {
                TritonCodegen::codegen_sort_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::histogram" => {
                TritonCodegen::codegen_histogram_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::reduce" => {
                TritonCodegen::codegen_reduce_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::associative_scan" => {
                TritonCodegen::codegen_associative_scan_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::atomic_add" => {
                TritonCodegen::codegen_atomic_add_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::atomic_max" => {
                TritonCodegen::codegen_atomic_max_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::atomic_min" => {
                TritonCodegen::codegen_atomic_min_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::atomic_xchg" => {
                TritonCodegen::codegen_atomic_xchg_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::atomic_cas" => {
                TritonCodegen::codegen_atomic_cas_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::atomic_and" => {
                TritonCodegen::codegen_atomic_and_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::atomic_or" => {
                TritonCodegen::codegen_atomic_or_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::atomic_xor" => {
                TritonCodegen::codegen_atomic_xor_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::umulhi" => {
                TritonCodegen::codegen_umulhi_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::rand" => {
                TritonCodegen::codegen_rand_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::randn" => {
                TritonCodegen::codegen_randn_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::randint" => {
                TritonCodegen::codegen_randint_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::randint4x" => {
                TritonCodegen::codegen_randint4x_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::inline_asm_elementwise" => {
                TritonCodegen::codegen_inline_asm_elementwise_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::multiple_of" => {
                TritonCodegen::codegen_multiple_of_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::max_contiguous" => {
                TritonCodegen::codegen_max_contiguous_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::max_constancy" => {
                TritonCodegen::codegen_max_constancy_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::where_" => {
                TritonCodegen::codegen_where_call as LocalCallHandler<'a, 'tcx>
            }
            // `Into::into` for tensor types is an identity conversion at the MLIR level
            // (T::I32Tensor and T::Tensor<i32> share the same MLIR type in LlvmTriton).
            // Pass the argument through unchanged so the SSA value retains its static shape.
            "Into::into" => TritonCodegen::codegen_identity_call as LocalCallHandler<'a, 'tcx>,
            "triton::Triton::assume" => {
                TritonCodegen::codegen_assume_call as LocalCallHandler<'a, 'tcx>
            }
            "triton::Triton::device_assert" => {
                TritonCodegen::codegen_device_assert_call as LocalCallHandler<'a, 'tcx>
            }
            // device_print is generic (has tensor<> params) so its monomorphization is
            // skipped by codegen_function; intercept the call here to avoid dangling tt.call.
            "triton::Triton::device_print" => {
                TritonCodegen::codegen_device_print_call as LocalCallHandler<'a, 'tcx>
            }
            _ => {
                eprintln!("[DEBUG] codegen_terminator_call: unhandled func_name={:?}", func_name);
                TritonCodegen::codegen_call as LocalCallHandler<'a, 'tcx>
            }
        };

        let value = method(
            self,
            tcx,
            instance,
            mir,
            func,
            func_name.as_str(),
            args,
            destination,
            target,
            unwind,
            call_source,
            fn_span,
            location,
            mlir_block,
            state,
        )?;

        if let Some(value) = value {
            state.ssa_values.insert(destination.local, value);
        }
        Ok(())
    }

    pub fn codegen_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        func: &Operand<'tcx>,
        func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // Pass-through for Range/ADT locals in tuple_fields (e.g., Range::into_iter).
        // This avoids crashing when codegen_operand tries to evaluate a Range aggregate local.
        if args.len() == 1 {
            if let Operand::Move(p) | Operand::Copy(p) = &args[0].node {
                if p.projection.is_empty() && destination.projection.is_empty() {
                    if let Some(fields) = state.tuple_fields.get(&p.local).cloned() {
                        state.tuple_fields.insert(destination.local, fields)?;
                        return Ok(None);
                    }
                }
            }
        }

        let args = args
            .iter()
            .map(|arg| {
                self.codegen_operand(
                    tcx,
                    instance,
                    &arg.node,
                    arg.node.ty(mir, tcx),
                    location,
                    mlir_block,
                    state,
                )
            })
            .collect::<Result<Vec<_>, MlirError>>()?;

        let fn_ty = func.ty(mir, tcx);
        let fn_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            ty::EarlyBinder::bind(tcx, fn_ty),
        );
        let ret_ty = match fn_ty.kind() {
            // Function definitions (e.g., closures and fn items).
            TyKind::FnDef(def_id, substs) => {
                let sig = tcx
                    .fn_sig(*def_id)
                    .instantiate(tcx, substs.no_bound_vars().unwrap())
                    .skip_norm_wip();
                sig.output().skip_binder()
            }
            // For function pointers, combine binder + header and get the output type.
            TyKind::FnPtr(binder, header) => {
                let full_sig = binder.with(*header);
                full_sig.output().skip_binder()
            }
            // Otherwise, fallback to just using the type itself.
            _ => fn_ty,
        };

        // Get the callee from the func operand: this should be the mangled function name.
        // This branch is for both `TyKind::FnDef` and function pointer cases.
        let callee_name = match func {
            Operand::Constant(constant) => {
                let ty = constant.const_.ty();
                let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    TypingEnv::fully_monomorphized(),
                    EarlyBinder::bind(tcx, ty),
                );
                match ty.kind() {
                    TyKind::FnDef(def_id, substs) => {
                        let typing_env = TypingEnv::post_analysis(tcx, *def_id);
                        if let Some(instance) =
                            Instance::resolve_for_fn_ptr(
                                tcx,
                                typing_env,
                                *def_id,
                                substs.no_bound_vars().unwrap(),
                            )
                        {
                            tcx.symbol_name(instance).name.to_string()
                        } else {
                            func_name.to_string()
                        }
                    }
                    TyKind::FnPtr(binder, header) => {
                        todo!("FnPtr: {:?} {:?}", binder, header);
                    }
                    _ => func_name.to_string(),
                }
            }
            // Try to resolve the function pointer to a DefId if possible.
            // Most common "direct call" case handled above, fallback to func_name param.
            _ => func_name.to_string(),
        };

        // Flatten the return type: unit → [], tuple → multiple types, scalar → one type.
        let result_types: Vec<_> = if ret_ty.is_unit() {
            vec![]
        } else if let TyKind::Tuple(elem_tys) = ret_ty.kind() {
            elem_tys
                .iter()
                .map(|ty| self.type_mapper.map_type(self.module.context(), &tcx, &ty))
                .collect()
        } else {
            vec![self.type_mapper.map_type(self.module.context(), &tcx, &ret_ty)]
        };

        let call_op: Operation<'a> =
            call(self.module.context(), location, callee_name.as_str(), &args, &result_types)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();

        let result = match result_types.len() {
            0 => None,
            1 => {
                let result = call_op.result(0).expect("Call operation result not found");
                Some(result.into())
            }
            _ => {
                // Multiple return values — caller must route them into tuple_fields.
                // Return the first value here; the tuple routing is done by the caller.
                let result = call_op.result(0).expect("Call operation result not found");
                Some(result.into())
            }
        };

        mlir_block.append_operation(call_op);
        Ok(result)
    }

    /// Identity pass-through for `Into::into` on tensor types.
    /// `T::I32Tensor` and `T::Tensor<i32>` are the same MLIR type in `LlvmTriton`;
    /// returning the argument unchanged preserves its static shape in the SSA.
    #[allow(clippy::too_many_arguments)]
    pub fn codegen_identity_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        debug_assert!(args.len() == 1, "codegen_identity_call: expected 1 arg");
        let val = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        Ok(Some(val))
    }

    #[allow(clippy::too_many_arguments)]
    fn codegen_switch_int<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        discr: &Operand<'tcx>,
        targets: &SwitchTargets,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        basic_blocks: &HashMap<BasicBlock, BlockRef<'a, 'a>>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        // Constant-fold SwitchInt when the discriminant is statically known.
        // This avoids emitting cf.cond_br referencing pruned dead MLIR blocks.
        use crate::mlir::codegen::triton::extract_switch_const;
        if let Some(const_val) =
            extract_switch_const(tcx, instance, discr, &state.const_disc_locals)
        {
            let target = targets
                .iter()
                .find(|(val, _)| *val == const_val as u128)
                .map(|(_, bb)| bb)
                .unwrap_or_else(|| targets.otherwise());
            return self.codegen_goto(location, &target, mlir_block, basic_blocks, state);
        }

        let discr_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, discr.ty(mir, tcx)),
        );

        let discr_value =
            self.codegen_operand(tcx, instance, discr, discr_ty, location, mlir_block, state)?;

        let ctx = self.module.context();
        let otherwise_bb = targets.otherwise();
        let cases: Vec<(u128, BasicBlock)> = targets.iter().collect();

        match cases.as_slice() {
            [] => self.codegen_goto(location, &otherwise_bb, mlir_block, basic_blocks, state),
            [(val, target_bb)] => self.codegen_switch_binary(
                discr_value,
                *val,
                *target_bb,
                otherwise_bb,
                location,
                mlir_block,
                basic_blocks,
                state,
            ),
            // Two explicit cases whose declared `otherwise` arm exists only for
            // exhaustiveness, not because it's reachable. rustc emits this for a fully
            // -covered 2-variant discriminant (e.g. `Option`'s `Iterator::next()` match)
            // whenever it doesn't fold the match down to a single-case switch itself --
            // observed as a real MIR-shape difference between rustc versions specifically
            // for the nvptx64-nvidia-cuda target (the same source produces a single-case
            // switch on other targets). Since `otherwise`'s block body is just
            // `unreachable`, the switch is really binary: not-val0 implies val1. Reuse the
            // single-case lowering with target1 standing in for "otherwise".
            [(val0, target0), (_val1, target1)]
                if self.is_unreachable_block(mir, otherwise_bb) =>
            {
                self.codegen_switch_binary(
                    discr_value,
                    *val0,
                    *target0,
                    *target1,
                    location,
                    mlir_block,
                    basic_blocks,
                    state,
                )
            }
            _ => todo!("SwitchInt with {} cases: {:?}", cases.len(), targets),
        }
    }

    /// True if `bb`'s body is just `unreachable` -- i.e. rustc has already proven this
    /// block can never execute (typically the formal `otherwise` arm of a `SwitchInt` over
    /// a discriminant whose declared cases are already exhaustive).
    fn is_unreachable_block(&self, mir: &Body<'_>, bb: BasicBlock) -> bool {
        matches!(mir.basic_blocks[bb].terminator().kind, TerminatorKind::Unreachable)
    }

    /// Lowers a binary discriminant check (`discr == val ? target_bb : otherwise_bb`) to
    /// `cf.cond_br`. Shared by the true
    /// single-explicit-case `SwitchInt` and the two-explicit-cases-with-unreachable
    /// -otherwise pattern, which differ only in how `otherwise_bb` was derived.
    #[allow(clippy::too_many_arguments)]
    fn codegen_switch_binary(
        &self,
        discr_value: Value<'a, 'a>,
        val: u128,
        target_bb: BasicBlock,
        otherwise_bb: BasicBlock,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        basic_blocks: &HashMap<BasicBlock, BlockRef<'a, 'a>>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        let ctx = self.module.context();

        let discr_mlir_ty = discr_value.r#type();
        let is_bool = IntegerType::try_from(discr_mlir_ty).is_ok_and(|ty| ty.width() == 1);
        let (cmp_result, target_bb, otherwise_bb) = if is_bool {
            // A switch on a bool branches on the bool itself rather than on `b == val`. A loop
            // lifted from these branches then guards its body with an `scf.if` on the same value
            // as its `scf.condition`, which is what lets canonicalization move the body into the
            // loop's `after` region and uplift a counted loop to `scf.for` (teenyc-3a1).
            if val == 0 {
                (discr_value, otherwise_bb, target_bb)
            } else {
                (discr_value, target_bb, otherwise_bb)
            }
        } else {
            // Emit: %const = arith.constant val : T
            //        %cmp  = arith.cmpi eq, %discr, %const : T
            //        cf.cond_br %cmp, ^target_bb, ^otherwise_bb
            let val_attr = IntegerAttribute::new(discr_mlir_ty, val as i64);
            let const_op: Operation<'a> =
                melior::dialect::arith::constant(ctx, val_attr.into(), location).into();
            let val_const: Value<'a, 'a> = const_op.result(0).expect("switch const").into();
            mlir_block.append_operation(const_op);

            let i1_ty = IntegerType::new(ctx, 1).into();
            let cmp_op: Operation<'a> =
                create_cmpi(ctx, location, Predicate::EQ, discr_value, val_const, i1_ty)
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into();
            let cmp_result: Value<'a, 'a> = cmp_op.result(0).expect("cmpi result").into();
            mlir_block.append_operation(cmp_op);
            (cmp_result, target_bb, otherwise_bb)
        };

        // Degenerate: both arms go to the same block (optimizer merged identical paths).
        // Emit cf.br instead to avoid Triton's makeTTGIR crashing on a cond_br with
        // two identical successors.
        if target_bb == otherwise_bb {
            return self.codegen_goto(location, &target_bb, mlir_block, basic_blocks, state);
        }

        let true_block = *basic_blocks.get(&target_bb).expect("switch target block");
        let false_block = *basic_blocks.get(&otherwise_bb).expect("switch otherwise block");

        let true_phi_args =
            self.branch_operands(target_bb, true_block, mlir_block, state, location)?;
        let false_phi_args =
            self.branch_operands(otherwise_bb, false_block, mlir_block, state, location)?;

        let cond_br_op = melior::dialect::cf::cond_br(
            ctx,
            cmp_result,
            &*true_block,
            &*false_block,
            &true_phi_args,
            &false_phi_args,
            location,
        );
        mlir_block.append_operation(cond_br_op);
        Ok(())
    }

    pub(crate) fn codegen_goto(
        &self,
        location: Location<'a>,
        target: &BasicBlock,
        mlir_block: &BlockRef<'a, 'a>,
        basic_blocks: &HashMap<BasicBlock, BlockRef<'a, 'a>>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(), MlirError> {
        let target_block = *basic_blocks.get(target).unwrap_or_else(|| {
            panic!(
                "codegen_goto: target {:?} not in basic_blocks (known: {:?})",
                target,
                basic_blocks.keys().collect::<Vec<_>>()
            )
        });

        let phi_args = self.branch_operands(*target, target_block, mlir_block, state, location)?;

        let br_op: Operation<'a> = if phi_args.is_empty() {
            create_cf_br(self.module.context(), location, &*target_block)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into()
        } else {
            melior::dialect::cf::br(&*target_block, &phi_args, location)
        };

        mlir_block.append_operation(br_op);
        Ok(())
    }

}

impl<'a> TritonCodegen<'a> {
    /// The operands of a branch from `block` to `target`: the value of each local that `target`
    /// merges, in the order of its block arguments.
    ///
    /// The first branch lowered into a block fixes its arguments, typed like the values it
    /// carries, so a tensor gets its concrete shape rather than the generic one its MIR type
    /// maps to. A merged local takes one argument per SSA value: one for a plain value, one per
    /// field of a tuple, and an `i1` "is `Some`" flag plus the payload for an Option. A merged
    /// local with no value on that first branch gets none. In reverse postorder that first branch
    /// precedes the block itself, so its arguments exist by the time it is lowered.
    pub(crate) fn branch_operands(
        &self,
        target: BasicBlock,
        target_block: BlockRef<'a, 'a>,
        block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
        location: Location<'a>,
    ) -> Result<Vec<Value<'a, 'a>>, MlirError> {
        let Some(phi_locals) = state.phi_join_locals.get(&target).cloned() else {
            return Ok(Vec::new());
        };
        let first_branch = state.phi_blocks_entered.insert(target);
        let mut operands = Vec::new();
        for local in phi_locals {
            if first_branch {
                let Some(values) = self.merged_values(local, block, state, location, None)? else {
                    continue;
                };
                let arg = match values {
                    PhiArg::Value(value) => {
                        operands.push(value);
                        PhiArg::Value(target_block.add_argument(value.r#type(), location))
                    }
                    PhiArg::Option { is_some, payload } => {
                        operands.extend([is_some, payload]);
                        PhiArg::Option {
                            is_some: target_block.add_argument(is_some.r#type(), location),
                            payload: target_block.add_argument(payload.r#type(), location),
                        }
                    }
                    PhiArg::Tuple(fields) => {
                        operands.extend(&fields);
                        PhiArg::Tuple(
                            fields
                                .iter()
                                .map(|field| target_block.add_argument(field.r#type(), location))
                                .collect(),
                        )
                    }
                };
                state.phi_block_args.insert((target, local), arg);
            } else if let Some(arg) = state.phi_block_args.get(&(target, local)).cloned() {
                let values = self.merged_values(local, block, state, location, Some(&arg))?;
                match (values, &arg) {
                    (Some(PhiArg::Value(value)), PhiArg::Value(_)) => operands.push(value),
                    (Some(PhiArg::Option { is_some, payload }), PhiArg::Option { .. }) => {
                        operands.extend([is_some, payload]);
                    }
                    (Some(PhiArg::Tuple(fields)), PhiArg::Tuple(arg_fields))
                        if fields.len() == arg_fields.len() =>
                    {
                        operands.extend(fields);
                    }
                    (None, _) => {
                        return Err(MlirError::CodegenFailed {
                            err: format!(
                                "{local:?} has no value on this branch to {target:?}, which \
                                 merges it from another path"
                            ),
                        });
                    }
                    _ => {
                        return Err(MlirError::CodegenFailed {
                            err: format!(
                                "{local:?} has a different kind of value on this branch to \
                                 {target:?} than on another path into it"
                            ),
                        });
                    }
                }
            }
        }
        Ok(operands)
    }

    /// A zero of `ty`, standing in for the payload of a `None` Option `local` at a join (it is
    /// never read: the merged `is_some` flag is false on that path).
    fn placeholder_payload(
        &self,
        local: Local,
        ty: melior::ir::Type<'a>,
        block: &BlockRef<'a, 'a>,
        location: Location<'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let ctx = self.module.context();
        let scalar_zero = |ty: melior::ir::Type<'a>| -> Option<melior::ir::Attribute<'a>> {
            if ty.is_integer() {
                Some(IntegerAttribute::new(ty, 0).into())
            } else if ty.is_float() {
                Some(FloatAttribute::new(ctx, ty, 0.0).into())
            } else {
                None
            }
        };
        let attr = match RankedTensorType::try_from(ty) {
            Ok(tensor) => scalar_zero(tensor.element())
                .and_then(|zero| DenseElementsAttribute::new(ty, &[zero]).ok().map(Into::into)),
            Err(_) => scalar_zero(ty),
        };
        let attr = attr.ok_or_else(|| MlirError::CodegenFailed {
            err: format!(
                "{local:?} is `None` on one path and `Some` on another, but its payload type \
                 ({ty}) has no placeholder value to merge them with"
            ),
        })?;
        let op = melior::dialect::arith::constant(ctx, attr, location);
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        block.append_operation(op);
        Ok(result.into())
    }

    /// The SSA values of `local` at the end of `block`, as they are passed to a join. A `None`
    /// Option passes `false` and a zero placeholder payload, of `arg`'s payload type when the
    /// join's arguments are already fixed.
    fn merged_values(
        &self,
        local: Local,
        block: &BlockRef<'a, 'a>,
        state: &CodegenState<'a, 'a>,
        location: Location<'a>,
        arg: Option<&PhiArg<'a, 'a>>,
    ) -> Result<Option<PhiArg<'a, 'a>>, MlirError> {
        if let Some(&value) = state.ssa_values.get(&local) {
            return Ok(Some(PhiArg::Value(value)));
        }
        if let Some(fields) = state.tuple_fields.get(&local) {
            return Ok(Some(PhiArg::Tuple(fields.clone())));
        }
        let Some(&option) = state.option_table.get(&local) else { return Ok(None) };
        let ctx = self.module.context();
        let flag = |value: bool| -> Result<Value<'a, 'a>, MlirError> {
            let op = melior::dialect::arith::constant(
                ctx,
                IntegerAttribute::new(IntegerType::new(ctx, 1).into(), i64::from(value)).into(),
                location,
            );
            let result =
                op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
            block.append_operation(op);
            Ok(result.into())
        };
        let (is_some, payload) = match option {
            OptionValue::Some(payload) => (flag(true)?, payload),
            OptionValue::Merged { is_some, payload } => (is_some, payload),
            OptionValue::None => {
                let payload_ty = match arg {
                    Some(PhiArg::Option { payload, .. }) => payload.r#type(),
                    _ => *state.phi_option_payload_types.get(&local).ok_or_else(|| {
                        MlirError::CodegenFailed {
                            err: format!(
                                "{local:?} is `None` on one path and `Some` on another, and its \
                                 payload type isn't known where the paths meet (a tensor Option \
                                 must be `Some` on the first path lowered into the join)"
                            ),
                        }
                    })?,
                };
                (flag(false)?, self.placeholder_payload(local, payload_ty, block, location)?)
            }
        };
        Ok(Some(PhiArg::Option { is_some, payload }))
    }
}
