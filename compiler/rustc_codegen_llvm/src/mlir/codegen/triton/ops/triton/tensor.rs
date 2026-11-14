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

use std::collections::HashSet;

use melior::ir::attribute::FloatAttribute;
use melior::ir::operation::{OperationBuilder, OperationLike};
use melior::ir::r#type::IntegerType;
use melior::ir::r#type::RankedTensorType;
use melior::ir::{
    Block, BlockLike, BlockRef, Location, Operation, Region, RegionLike, ShapedTypeLike, TypeLike,
    Value, ValueLike,
};
use rustc_ast::{FloatTy, IntTy};
use rustc_middle::mir::interpret::Scalar;
use rustc_middle::mir::{
    BasicBlock, Body, CallSource, CastKind, ConstValue, Local, Operand, Place, RETURN_PLACE, Rvalue,
    START_BLOCK, StatementKind, TerminatorKind, UnwindAction,
};
use rustc_middle::ty::adjustment::PointerCoercion;
use rustc_middle::ty::{EarlyBinder, Instance, Ty, TyCtxt, TyKind, TypingEnv};
use rustc_mlir::shared::arith::{Int, create_extsi, create_int_constant};
use rustc_mlir::shared::builtin::tensor_type;
use rustc_mlir::triton::program::{ProgramAxis, create_get_num_programs};
use rustc_target::spec::Arch;
use rustc_mlir::triton::tensor::{
    CacheModifier, EvictionPolicy, InputPrecision, MemSemantic, MemSyncScope, PropagateNan, RmwOp,
    ScaleDotElemType, add_ptr, assert_op, atomic_cas, atomic_rmw, broadcast, clampf,
    descriptor_load, descriptor_store, dot, dot_scaled, expand_dims, gather, histogram, join, load,
    make_range, make_tensor_descriptor, mulhiui, precise_divf, precise_sqrt,
    print as triton_print, reduce, reduce_return, reshape, scan, scan_return, splat, split, store,
    trans, bitcast,
};
use rustc_mlir::triton::{int_to_ptr, pointer_type};
use rustc_span::Span;
use rustc_span::Spanned;

use crate::mlir::codegen::triton::location::span_to_location;
use crate::mlir::codegen::triton::{BlockPtr, CodegenState, TritonCodegen};
use crate::mlir::errors::MlirError;

/// Which extreme an index reduction keeps.
#[derive(Clone, Copy, Debug)]
enum Extremum {
    Max,
    Min,
}

/// The value comparisons of an index-reduction combine, chosen from the element type:
/// ordered float compares, or signed/unsigned integer compares (MLIR integers are signless,
/// so signedness comes from the Rust element type).
#[derive(Clone, Copy)]
enum ValueCompare {
    Float(melior::dialect::arith::CmpfPredicate),
    Int(melior::dialect::arith::CmpiPredicate),
}

impl ValueCompare {
    fn for_elem(rust_elem: &TyKind<'_>, elem: melior::ir::Type<'_>, extremum: Extremum) -> Self {
        use melior::dialect::arith::{CmpfPredicate as F, CmpiPredicate as I};
        match (rust_elem, extremum) {
            (TyKind::Uint(_), Extremum::Max) => Self::Int(I::Ugt),
            (TyKind::Uint(_), Extremum::Min) => Self::Int(I::Ult),
            (TyKind::Int(_), Extremum::Max) => Self::Int(I::Sgt),
            (TyKind::Int(_), Extremum::Min) => Self::Int(I::Slt),
            _ if elem.is_integer() || elem.is_index() => match extremum {
                Extremum::Max => Self::Int(I::Sgt),
                Extremum::Min => Self::Int(I::Slt),
            },
            (_, Extremum::Max) => Self::Float(F::Ogt),
            (_, Extremum::Min) => Self::Float(F::Olt),
        }
    }

    /// `lhs` is strictly better than `rhs` (greater for max, less for min).
    fn better<'c>(
        self,
        context: &'c melior::Context,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Operation<'c> {
        match self {
            Self::Float(p) => melior::dialect::arith::cmpf(context, p, lhs, rhs, location),
            Self::Int(p) => melior::dialect::arith::cmpi(context, p, lhs, rhs, location),
        }
    }

    fn equal<'c>(
        self,
        context: &'c melior::Context,
        lhs: Value<'c, '_>,
        rhs: Value<'c, '_>,
        location: Location<'c>,
    ) -> Operation<'c> {
        use melior::dialect::arith::{CmpfPredicate, CmpiPredicate, cmpf, cmpi};
        match self {
            Self::Float(_) => cmpf(context, CmpfPredicate::Oeq, lhs, rhs, location),
            Self::Int(_) => cmpi(context, CmpiPredicate::Eq, lhs, rhs, location),
        }
    }
}

/// Fixed parameters of one bitonic sort (see `codegen_sort_call`).
#[derive(Clone, Copy)]
struct Bitonic<'a> {
    /// The hypercube's rank, `log2(numel)`.
    n_dims: u32,
    elem: melior::ir::Type<'a>,
    /// The same-width integer type the partner exchange XORs in.
    int_elem: melior::ir::Type<'a>,
    compare: ValueCompare,
    location: Location<'a>,
}

/// The `order` argument of Triton's `_bitonic_merge_hypercube`.
#[derive(Clone, Copy)]
enum BitonicOrder {
    Alternating,
    Fixed { descending: bool },
}

/// Triton's `flip` in `_compare_and_swap`: a constant 0 or 1, or the indicator of a stage.
#[derive(Clone, Copy)]
enum Flip {
    Zero,
    One,
    Indicator(u32),
}

impl<'a> TritonCodegen<'a> {
    /// `arith.constant 0 : <ty>` for a scalar integer or float type, at that exact type.
    ///
    /// Integer zeros must be built at the element's own width: splatting an `i32` zero into a
    /// `tensor<..xi8>` (or subtracting from an `i16`) fails MLIR verification with "requires the
    /// same element type for all operands and results" (teenygrad#142).
    pub(crate) fn zero_constant(
        &self,
        location: Location<'a>,
        ty: melior::ir::Type<'a>,
    ) -> Operation<'a> {
        let attr: melior::ir::Attribute<'a> = if ty.is_integer() {
            melior::ir::attribute::IntegerAttribute::new(ty, 0).into()
        } else {
            FloatAttribute::new(self.module.context(), ty, 0.0).into()
        };
        melior::dialect::arith::constant(self.module.context(), attr, location)
    }

    pub fn codegen_arange<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        func: &Operand<'tcx>,
        _func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        target: &Option<BasicBlock>,
        unwind: &UnwindAction,
        call_source: &CallSource,
        fn_span: &Span,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        _state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        println!(
            "[DEBUG] TritonCodegen::codegen_arange: func: {:?} args: {:?} destination: {:?} target: {:?} unwind: {:?} call_source: {:?} fn_span: {:?}",
            func, args, destination, target, unwind, call_source, fn_span
        );

        debug_assert!(
            args.len() == 2,
            "TritonCodegen::codegen_arange: args length must be 2: {:?}",
            args
        );

        // Use try_eval_const_i32 so that computed const-generic expressions like
        // `HEAD_DIM / 2` (which emit a MIR local rather than a literal constant)
        // are constant-folded before being passed to the Triton make_range op.
        let start = self.try_eval_const_i32(tcx, instance, mir, &args[0].node)?;
        let end = self.try_eval_const_i32(tcx, instance, mir, &args[1].node)?;

        let arange_op: Operation<'a> = make_range(self.module.context(), location, start, end)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();

        let result = arange_op.result(0).expect("Arange operation result not found");
        eprintln!("[DEBUG] AXM TritonCodegen::codegen_arange: {:?}", arange_op.to_string());
        mlir_block.append_operation(arange_op);
        Ok(Some(result.into()))
    }

    /// `Triton::arange_f32(start, end)` — like `arange` but returns a 1-D `f32`
    /// tensor `[start as f32, (start+1) as f32, …, (end-1) as f32]`.
    ///
    /// Implemented as `make_range` followed by `arith.sitofp`, avoiding the
    /// MIR `copy` nodes that would arise from casting an intermediate i32 local.
    pub fn codegen_arange_f32<'tcx>(
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
        _state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        debug_assert!(
            args.len() == 2,
            "TritonCodegen::codegen_arange_f32: args length must be 2: {:?}",
            args
        );

        let start = self.try_eval_const_i32(tcx, instance, mir, &args[0].node)?;
        let end = self.try_eval_const_i32(tcx, instance, mir, &args[1].node)?;
        let n = (end - start) as i64;

        // Create i32 range [start, end).
        let arange_op: Operation<'a> = make_range(self.module.context(), location, start, end)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let i32_range: Value<'a, 'a> = arange_op.result(0).expect("arange_f32: make_range").into();
        mlir_block.append_operation(arange_op);

        // Cast i32 tensor → f32 tensor via arith.sitofp.
        let f32_ty = melior::ir::Type::float32(self.module.context());
        let f32_tensor_ty = tensor_type(&[n], f32_ty).into();
        let cast_op: Operation<'a> =
            melior::dialect::arith::sitofp(i32_range, f32_tensor_ty, location).into();
        let f32_range: Value<'a, 'a> = cast_op.result(0).expect("arange_f32: sitofp").into();
        mlir_block.append_operation(cast_op);

        Ok(Some(f32_range))
    }

    pub fn codegen_add_ptr<'tcx>(
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
        debug_assert!(
            args.len() == 2,
            "TritonCodegen::codegen_add_offsets_call: args length must be 2"
        );

        let arg0 = &args[0].node;
        let arg1 = &args[1].node;

        let ptr = self.codegen_operand(
            tcx,
            instance,
            arg0,
            arg0.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let offset = self.codegen_operand(
            tcx,
            instance,
            arg1,
            arg1.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        debug_assert!(
            offset.r#type().is_tensor(),
            "TritonCodegen::codegen_add_offset: rhs is not a tensor"
        );

        let ptr = self.like_tensor(tcx, location, offset, ptr, mlir_block)?;

        let add_ptr_op: Operation<'a> =
            add_ptr(self.module.context(), location, ptr, offset, ptr.r#type())
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let result = add_ptr_op.result(0).expect("AddPtr operation result not found");

        eprintln!("[DEBUG] AXM TritonCodegen::codegen_add_ptr: {:?}", add_ptr_op.to_string());
        mlir_block.append_operation(add_ptr_op);
        Ok(Some(result.into()))
    }

    pub fn codegen_load<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
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
        // args: [ptr_tensor, Option<mask>, Option<other>, &[i32], Option<PaddingOption>,
        //        Option<CacheModifier>, Option<EvictionPolicy>, bool]
        let ptr = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let mask = self.codegen_option_operand(
            tcx,
            instance,
            mir,
            &args[1].node,
            location,
            mlir_block,
            state,
        )?;
        let other = self.codegen_option_operand(
            tcx,
            instance,
            mir,
            &args[2].node,
            location,
            mlir_block,
            state,
        )?;
        // tt.load only accepts `other` alongside a mask: its verifier checks the mask slot
        // whenever there is more than one operand. Without a mask every lane is loaded, so
        // `other` can never be selected and is dropped rather than emitted.
        let other = if mask.is_some() { other } else { None };
        let mask = mask.map(|m| self.broadcast_to_ptr_shape(m, ptr, location, mlir_block)).transpose()?;
        let other =
            other.map(|o| self.broadcast_to_ptr_shape(o, ptr, location, mlir_block)).transpose()?;

        // Derive result type from the ptr operand's MLIR type.
        // ptr is `tensor<Nx!tt.ptr<T>>` — extract shape [N] and element type T from
        // the destination Rust type (which gives us `tensor<?xT>`; we take the element).
        let result_ty = {
            let dest_ty = destination.ty(mir, tcx).ty;
            let dest_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(tcx, dest_ty),
            );
            let dest_mlir_ty = self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty);

            if let Ok(ptr_tensor_ty) = RankedTensorType::try_from(ptr.r#type()) {
                // ptr is tensor<Nx!tt.ptr<ElemTy>> — use the ptr's concrete shape.
                let shape: Vec<i64> = ptr_tensor_ty
                    .dims()
                    .map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
                // Element type of result: unwrap outer tensor<> wrapper from dest_mlir_ty if present.
                let elem_ty = if let Ok(dest_tensor_ty) = RankedTensorType::try_from(dest_mlir_ty) {
                    dest_tensor_ty.element()
                } else {
                    dest_mlir_ty
                };
                tensor_type(&shape, elem_ty).into()
            } else {
                // Scalar pointer — result type is exactly the destination type.
                dest_mlir_ty
            }
        };

        let load_op: Operation<'a> = load(
            self.module.context(),
            location,
            ptr,
            mask,
            other,
            result_ty,
            CacheModifier::None,
            EvictionPolicy::Normal,
            false,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();

        let result =
            load_op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(load_op);
        Ok(Some(result.into()))
    }

    /// `Triton::load_scalar_f32_as_i32(ptr, offset)` — scalar gather with f32→i32 cast.
    ///
    /// 1. `tt.addptr(%ptr, %offset)` on scalar `!tt.ptr<f32>` → scalar `!tt.ptr<f32>`
    /// 2. `tt.load` → scalar `f32`
    /// 3. `arith.fptosi` → scalar `i32`
    ///
    /// Used when token IDs are stored as f32 (the default graph dtype).
    /// The returned `i32` is usable directly in scalar arithmetic such as
    /// `I32Tensor + i32` offset expressions.
    pub fn codegen_load_scalar_f32_as_i32<'tcx>(
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
        debug_assert!(args.len() == 2, "load_scalar_f32_as_i32: expected 2 args");

        let ptr = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let offset = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        // ptr + offset → !tt.ptr<f32>
        let f32_ty = melior::ir::Type::float32(self.module.context());
        let f32_ptr_ty = pointer_type(f32_ty);
        let addptr_op: Operation<'a> =
            add_ptr(self.module.context(), location, ptr, offset, f32_ptr_ty)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let new_ptr: Value<'a, 'a> =
            addptr_op.result(0).expect("load_scalar_f32_as_i32: addptr result").into();
        mlir_block.append_operation(addptr_op);

        // tt.load !tt.ptr<f32> → f32
        let load_op: Operation<'a> = load(
            self.module.context(),
            location,
            new_ptr,
            None,
            None,
            f32_ty,
            CacheModifier::None,
            EvictionPolicy::Normal,
            false,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        let f32_val: Value<'a, 'a> =
            load_op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?.into();
        mlir_block.append_operation(load_op);

        // arith.fptosi f32 → i32
        let i32_ty = IntegerType::new(self.module.context(), 32).into();
        let fptosi_op: Operation<'a> =
            melior::dialect::arith::fptosi(f32_val, i32_ty, location).into();
        let i32_val: Value<'a, 'a> =
            fptosi_op.result(0).expect("load_scalar_f32_as_i32: fptosi result").into();
        mlir_block.append_operation(fptosi_op);

        Ok(Some(i32_val))
    }

    /// Broadcast a store's value or mask, or a load's mask or `other`, to the shape of the
    /// pointer tensor `ptr`, as Triton's frontend does: `tt.store` and `tt.load` require them
    /// to match it, and storing a keep_dims reduction to every lane is the usual way to write
    /// one value per program. Only a same-rank tensor whose every dimension is 1 or equal to the
    /// pointer's is broadcast; anything else is returned unchanged for the verifier to report.
    fn broadcast_to_ptr_shape(
        &self,
        v: Value<'a, 'a>,
        ptr: Value<'a, 'a>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let (Ok(ptr_t), Ok(v_t)) =
            (RankedTensorType::try_from(ptr.r#type()), RankedTensorType::try_from(v.r#type()))
        else {
            return Ok(v);
        };
        let dims = |t: RankedTensorType<'a>| {
            t.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })
        };
        let (target, from) = (dims(ptr_t)?, dims(v_t)?);
        let broadcastable = from.len() == target.len()
            && from.iter().zip(&target).all(|(&f, &t)| f == 1 || f == t);
        if from == target || !broadcastable {
            return Ok(v);
        }
        let op: Operation<'a> =
            broadcast(self.module.context(), location, v, tensor_type(&target, v_t.element()).into())
                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
                .into();
        self.append_result(op, mlir_block)
    }

    pub fn codegen_store<'tcx>(
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
        // args: [ptr_tensor, value_tensor, Option<mask>, &[i32],
        //        Option<CacheModifier>, Option<EvictionPolicy>]
        let ptr = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let value = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let mask = self.codegen_option_operand(
            tcx,
            instance,
            mir,
            &args[2].node,
            location,
            mlir_block,
            state,
        )?;

        let value = self.broadcast_to_ptr_shape(value, ptr, location, mlir_block)?;
        let mask = mask.map(|m| self.broadcast_to_ptr_shape(m, ptr, location, mlir_block)).transpose()?;

        let store_op: Operation<'a> = store(
            self.module.context(),
            location,
            ptr,
            value,
            mask,
            CacheModifier::None,
            EvictionPolicy::Normal,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();

        mlir_block.append_operation(store_op);
        Ok(None)
    }

    pub fn codegen_maximum<'tcx>(
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
        // args: [lhs_tensor, rhs_tensor]
        let lhs = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let rhs = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        // arith.maximumf requires both operands to have the result type, so a keep-dims
        // reduction (tensor<1xf32>) against a full tile has to be broadcast first.
        let (lhs, rhs) = self.maybe_broadcast_pair(location, lhs, rhs, mlir_block)?;

        let op_name = self.choose_signed_op(
            tcx,
            instance,
            args[0].node.ty(mir, tcx),
            lhs,
            ["arith.maximumf", "arith.maxsi", "arith.maxui"],
        );
        let result_ty = lhs.r#type();
        let max_op: Operation<'a> = OperationBuilder::new(op_name, location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;

        let result =
            max_op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(max_op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::transmute`-as-slice: `transmute<(*const T, usize), &[T]>((ptr, len))`
    ///
    /// This is used in `no_core` to build a `&[i32]` shape descriptor from a constant array.
    /// We extract the shape from `state.const_arrays` / `state.ptr_to_const_array` and store it
    /// in `state.slice_shape`.  No MLIR operation is emitted.
    pub fn codegen_transmute_slice<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _instance: &Instance<'tcx>,
        _mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        func_name: &str,
        args: &[Spanned<Operand<'tcx>>],
        destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        _location: Location<'a>,
        _mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        eprintln!("[DEBUG] codegen_transmute_slice: {func_name}, args: {args:?}");

        // The single argument is the fat-pointer tuple `(*const T, usize)` whose first field
        // is a const-array pointer.  The Tuple aggregate handler already stored the shape in
        // `slice_shape` keyed by the tuple local.  Propagate it to the destination.
        if let Some(Spanned { node: Operand::Move(src_place) | Operand::Copy(src_place), .. }) =
            args.first()
        {
            if let Some(shape) = state.slice_shape.get(&src_place.local).cloned() {
                state.slice_shape.insert(destination.local, shape.clone())?;
                eprintln!(
                    "[DEBUG] codegen_transmute_slice: shape={shape:?} → {:?}",
                    destination.local
                );
                return Ok(None);
            }
            if let Some(vals) = state.slice_dyn_values.get(&src_place.local).cloned() {
                state.slice_dyn_values.insert(destination.local, vals)?;
                eprintln!("[DEBUG] codegen_transmute_slice: dyn_values → {:?}", destination.local);
                return Ok(None);
            }
        }

        eprintln!("[DEBUG] codegen_transmute_slice: could not extract shape; args={args:?}");
        Ok(None)
    }

    /// `triton::Triton::zeros` — creates a tensor filled with zeros.
    ///
    /// The shape is recovered from `state.slice_shape` (populated by `codegen_transmute_slice`).
    /// The element type comes from the MIR return-place type.
    pub fn codegen_zeros<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
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
        // The shape has to be known at compile time; a one-element fallback would compile a
        // kernel that silently works on the wrong number of elements.
        let shape_arg = &args
            .first()
            .ok_or_else(|| MlirError::CodegenFailed { err: "zeros: missing shape".to_string() })?
            .node;
        let shape = self.shape_from_slice_arg(shape_arg, state, "zeros")?;

        // Get the element Rust type from the destination.
        let dest_ty = destination.ty(mir, tcx).ty;
        let dest_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, dest_ty),
        );

        // Extract the element MLIR type:
        // - For `Tensor<D>` (ADT), map through the type mapper
        // - For direct scalar (unlikely), use directly
        let elem_mlir_ty = self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty);

        // Unwrap: if we got a tensor type back, extract its element type.
        let elem_mlir_ty = if elem_mlir_ty.is_tensor() {
            let tensor_ty: RankedTensorType<'a> = elem_mlir_ty
                .try_into()
                .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
            tensor_ty.element()
        } else {
            elem_mlir_ty
        };

        // For pointer element types: emit tt.splat(tt.int_to_ptr(0)) — a tensor of null ptrs.
        // Can't use arith.constant for pointer types (DenseIntOrFPElementsAttr doesn't support
        // them), and ub.poison causes crashes in later Triton passes.
        // NOTE: must use is_float() not string heuristics — "!tt.ptr<f32>" contains 'f'.
        if !elem_mlir_ty.is_integer() && !elem_mlir_ty.is_float() {
            // Build: %null_i64 = arith.constant 0 : i64
            let null_i64_op: Operation<'a> =
                create_int_constant(self.module.context(), location, Int::I64(0))
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into();
            let null_i64 = null_i64_op.result(0).expect("null i64").into();
            mlir_block.append_operation(null_i64_op);
            // Build: %null_ptr = tt.int_to_ptr %null_i64 : i64 -> <pointer_elem_ty>
            let null_ptr_op: Operation<'a> =
                int_to_ptr(self.module.context(), location, null_i64, elem_mlir_ty)
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into();
            let null_ptr = null_ptr_op.result(0).expect("null ptr").into();
            mlir_block.append_operation(null_ptr_op);
            // Build: %result = tt.splat %null_ptr : <pointer_elem_ty> -> tensor<N x <pointer_elem_ty>>
            let tensor_ty = tensor_type(&shape, elem_mlir_ty).into();
            let splat_op: Operation<'a> =
                splat(self.module.context(), location, null_ptr, tensor_ty)
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into();
            let result = splat_op.result(0).expect("null ptr tensor").into();
            mlir_block.append_operation(splat_op);
            return Ok(Some(result));
        }

        // Create a scalar zero constant of the element type.
        let zero_op: Operation<'a> = self.zero_constant(location, elem_mlir_ty);
        let zero_val = zero_op.result(0).expect("zero constant result");
        mlir_block.append_operation(zero_op);

        // Splat the scalar zero to a tensor of the requested shape.
        let tensor_ty = tensor_type(&shape, elem_mlir_ty).into();
        let splat_op: Operation<'a> =
            splat(self.module.context(), location, zero_val.into(), tensor_ty)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let result = splat_op.result(0).expect("splat result");
        mlir_block.append_operation(splat_op);

        Ok(Some(result.into()))
    }

    pub fn codegen_zeros_like<'tcx>(
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
        debug_assert!(
            args.len() == 1,
            "TritonCodegen::codegen_zeros_like: args length must be 1: {:?}",
            args
        );

        let arg0 = &args[0].node;
        let tensor = self.codegen_operand(
            tcx,
            instance,
            arg0,
            arg0.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        // Extract tensor type from the input tensor's MLIR type.
        let tensor_ty: RankedTensorType<'a> = tensor
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = tensor_ty.element();

        // For pointer element types: splat a null pointer (same logic as codegen_zeros).
        if !elem_ty.is_integer() && !elem_ty.is_float() {
            let null_i64_op: Operation<'a> =
                create_int_constant(self.module.context(), location, Int::I64(0))
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into();
            let null_i64 = null_i64_op.result(0).expect("null i64").into();
            mlir_block.append_operation(null_i64_op);
            let null_ptr_op: Operation<'a> =
                int_to_ptr(self.module.context(), location, null_i64, elem_ty)
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into();
            let null_ptr = null_ptr_op.result(0).expect("null ptr").into();
            mlir_block.append_operation(null_ptr_op);
            let splat_op: Operation<'a> =
                splat(self.module.context(), location, null_ptr, tensor.r#type())
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into();
            let result = splat_op.result(0).expect("null ptr tensor").into();
            mlir_block.append_operation(splat_op);
            return Ok(Some(result));
        }

        // Create a zero constant of the element type.
        let zero_op: Operation<'a> = self.zero_constant(location, elem_ty);
        let zero_val = zero_op.result(0).expect("zero constant result");
        mlir_block.append_operation(zero_op);

        // Splat to the same tensor type as the input.
        let splat_op: Operation<'a> =
            splat(self.module.context(), location, zero_val.into(), tensor.r#type())
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let result = splat_op.result(0).expect("splat result");
        mlir_block.append_operation(splat_op);

        Ok(Some(result.into()))
    }

    /// `triton::Triton::cat` — concatenate two tensors along their only axis.
    ///
    /// args[0] = lhs tensor, args[1] = rhs tensor, args[2] = can_reorder (bool, unused in MLIR)
    pub fn codegen_cat_call<'tcx>(
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
        use melior::ir::ShapedTypeLike;

        let lhs_arg = &args[0].node;
        let rhs_arg = &args[1].node;
        let lhs = self.codegen_operand(
            tcx,
            instance,
            lhs_arg,
            lhs_arg.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let rhs = self.codegen_operand(
            tcx,
            instance,
            rhs_arg,
            rhs_arg.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        // Compute result shape: sum of each dimension across lhs and rhs.
        let lhs_tensor: RankedTensorType<'a> = lhs
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let rhs_tensor: RankedTensorType<'a> = rhs
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let lhs_dims =
            lhs_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let rhs_dims =
            rhs_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;

        // For a 1D cat, result size = lhs_dim + rhs_dim.
        let result_dims: Vec<i64> =
            lhs_dims.iter().zip(rhs_dims.iter()).map(|(&l, &r)| l + r).collect();
        let elem_ty = lhs_tensor.element();
        let result_ty = tensor_type(&result_dims, elem_ty).into();

        use melior::ir::operation::OperationBuilder;
        let cat_op: Operation<'a> = OperationBuilder::new("tt.cat", location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
            .into();
        let result = cat_op.result(0).expect("cat result");
        mlir_block.append_operation(cat_op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::cast` — element-wise type cast for tensors.
    ///
    /// args[0] = input tensor, args[1] = rounding mode (optional enum), args[2] = saturate (bool)
    ///
    /// For the common case where src and dst element types are the same, returns the input as-is.
    /// For type conversions, emits the appropriate `arith` cast operation.
    pub fn codegen_cast_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
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
        use melior::ir::ShapedTypeLike;

        let arg0 = &args[0].node;
        let src_value = self.codegen_operand(
            tcx,
            instance,
            arg0,
            arg0.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        // Determine the destination element type from the destination place type.
        let dest_rust_ty = destination.ty(mir, tcx).ty;
        let dest_rust_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, dest_rust_ty),
        );
        let dest_mlir_ty = self.type_mapper.map_type(self.module.context(), &tcx, &dest_rust_ty);

        // If destination type is dynamic (unknown-shape tensor), use the input's actual type.
        // Triton cast is always shape-preserving; the function signature just doesn't know
        // the shape statically.
        let dest_mlir_ty = if dest_mlir_ty.is_tensor() {
            let is_dynamic = dest_mlir_ty
                .try_into()
                .ok()
                .and_then(|t: RankedTensorType<'a>| t.dims().ok())
                .map(|dims| dims.iter().any(|&d| d == i64::MIN))
                .unwrap_or(true);
            if is_dynamic {
                // Construct target type: same shape as input, but with the destination element type
                let src_tensor: RankedTensorType<'a> =
                    src_value.r#type().try_into().map_err(|e: melior::error::Error| {
                        MlirError::InvalidType { msg: e.to_string() }
                    })?;
                let src_dims =
                    src_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
                // Get destination element type from the generic tensor handler result
                let dest_elem_ty = {
                    let mapped = dest_mlir_ty
                        .try_into()
                        .map(|t: RankedTensorType<'a>| t.element())
                        .unwrap_or(src_tensor.element());
                    mapped
                };
                tensor_type(&src_dims, dest_elem_ty).into()
            } else {
                dest_mlir_ty
            }
        } else {
            dest_mlir_ty
        };

        // If source and destination MLIR types are the same, the cast is a no-op.
        if src_value.r#type() == dest_mlir_ty {
            return Ok(Some(src_value));
        }

        // Otherwise emit the appropriate arith conversion op.
        let src_elem_ty = if src_value.r#type().is_tensor() {
            src_value
                .r#type()
                .try_into()
                .map(|t: RankedTensorType<'a>| t.element())
                .unwrap_or(src_value.r#type())
        } else {
            src_value.r#type()
        };
        let dst_elem_ty = if dest_mlir_ty.is_tensor() {
            dest_mlir_ty
                .try_into()
                .map(|t: RankedTensorType<'a>| t.element())
                .unwrap_or(dest_mlir_ty)
        } else {
            dest_mlir_ty
        };

        // melior typed_unary_operations signature: (in_value, out_type, location)
        let cast_op: Operation<'a> = if src_elem_ty.is_float() && dst_elem_ty.is_float() {
            // Float → float: extf (widening) or truncf-family (narrowing).
            // Determine by comparing bit widths via the string representation.
            // f16 < f32 < f64
            let float_bits = |ty: melior::ir::Type<'a>| -> u32 {
                let s = ty.to_string();
                if s.contains("f16") {
                    16
                } else if s.contains("f32") {
                    32
                } else if s.contains("f64") {
                    64
                } else {
                    128
                }
            };
            if float_bits(dst_elem_ty) < float_bits(src_elem_ty) {
                // Narrowing: use ODS builder directly since the melior wrapper
                // for truncf is only a same-type unary; fall back to OperationBuilder.
                use melior::ir::operation::OperationBuilder;
                OperationBuilder::new("arith.truncf", location)
                    .add_operands(&[src_value])
                    .add_results(&[dest_mlir_ty])
                    .build()
                    .map_err(|e| MlirError::CreateOperation {
                        err: rustc_mlir::errors::Error::IncompatibleTypes {
                            lhs: e.to_string(),
                            rhs: String::new(),
                        },
                    })?
                    .into()
            } else {
                // Widening
                melior::dialect::arith::extf(src_value, dest_mlir_ty, location).into()
            }
        } else if src_elem_ty.is_integer() && dst_elem_ty.is_float() {
            if Self::is_unsigned(tcx, instance, arg0.ty(mir, tcx)) {
                melior::dialect::arith::uitofp(src_value, dest_mlir_ty, location).into()
            } else {
                melior::dialect::arith::sitofp(src_value, dest_mlir_ty, location).into()
            }
        } else if src_elem_ty.is_float() && dst_elem_ty.is_integer() {
            if Self::is_unsigned(tcx, instance, dest_rust_ty) {
                melior::dialect::arith::fptoui(src_value, dest_mlir_ty, location).into()
            } else {
                melior::dialect::arith::fptosi(src_value, dest_mlir_ty, location).into()
            }
        } else {
            let width = |ty: melior::ir::Type<'a>| {
                IntegerType::try_from(ty)
                    .map(|t| t.width())
                    .map_err(|e| MlirError::InvalidType { msg: e.to_string() })
            };
            // Equal widths were returned above as a no-op, since the types are signless.
            if width(dst_elem_ty)? < width(src_elem_ty)? {
                melior::dialect::arith::trunci(src_value, dest_mlir_ty, location).into()
            } else if Self::is_unsigned(tcx, instance, arg0.ty(mir, tcx)) {
                melior::dialect::arith::extui(src_value, dest_mlir_ty, location).into()
            } else {
                melior::dialect::arith::extsi(src_value, dest_mlir_ty, location).into()
            }
        };
        let result = cast_op.result(0).expect("cast result");
        mlir_block.append_operation(cast_op);
        Ok(Some(result.into()))
    }

    // -------------------------------------------------------------------------
    // Helper: extract a shape Vec<i64> from a `&[i32]` slice operand tracked in
    // `state.slice_shape`.
    // -------------------------------------------------------------------------
    fn shape_from_slice_arg<'tcx>(
        &self,
        arg: &Operand<'tcx>,
        state: &CodegenState<'a, 'a>,
        ctx: &str,
    ) -> Result<Vec<i64>, MlirError> {
        match arg {
            Operand::Copy(p) | Operand::Move(p) => {
                state.slice_shape.get(&p.local).cloned().ok_or_else(|| MlirError::CodegenFailed {
                    err: format!("{ctx}: slice_shape not found for {:?}", p.local),
                })
            }
            other => Err(MlirError::CodegenFailed {
                err: format!("{ctx}: unexpected operand for shape arg: {other:?}"),
            }),
        }
    }

    /// Like `shape_from_slice_arg` but returns `Vec<i32>`, checking both `slice_shape` and
    /// `const_arrays` (the block_shape arg to `make_tensor_descriptor` is `&[i32]`).
    fn shape_from_slice_arg_i32<'tcx>(
        &self,
        arg: &Operand<'tcx>,
        state: &CodegenState<'a, 'a>,
    ) -> Result<Vec<i32>, MlirError> {
        match arg {
            Operand::Copy(p) | Operand::Move(p) => {
                if let Some(shape) = state.slice_shape.get(&p.local) {
                    return Ok(shape.iter().map(|&v| v as i32).collect());
                }
                Err(MlirError::CodegenFailed {
                    err: format!(
                        "shape_from_slice_arg_i32: slice_shape not found for {:?}",
                        p.local
                    ),
                })
            }
            other => Err(MlirError::CodegenFailed {
                err: format!("shape_from_slice_arg_i32: unexpected operand: {other:?}"),
            }),
        }
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::broadcast(a, b)` — broadcast two tensors to a common
    // shape.  Returns `(Tensor<D>, Tensor<D>)` stored in `tuple_fields`.
    // -------------------------------------------------------------------------
    pub fn codegen_broadcast_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
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
        let lhs_arg = &args[0].node;
        let rhs_arg = &args[1].node;
        let lhs = self.codegen_operand(
            tcx,
            instance,
            lhs_arg,
            lhs_arg.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let rhs = self.codegen_operand(
            tcx,
            instance,
            rhs_arg,
            rhs_arg.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        // Compute broadcast shape: if both tensors have identical shape, no ops needed.
        // For dynamic tensors, emit tt.broadcast to the broadened shape.
        // As a first approximation, broadcast both to the larger of the two shapes.
        let lhs_tensor: RankedTensorType<'a> = lhs
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let rhs_tensor: RankedTensorType<'a> = rhs
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let lhs_dims =
            lhs_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let rhs_dims =
            rhs_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;

        let broadcast_dims: Vec<i64> = lhs_dims
            .iter()
            .zip(rhs_dims.iter())
            .map(|(&l, &r)| {
                if l == r || r < 0 {
                    l
                } else if l < 0 {
                    r
                } else {
                    l.max(r)
                }
            })
            .collect();

        let elem_ty = lhs_tensor.element();
        let result_ty = tensor_type(&broadcast_dims, elem_ty).into();

        // Emit tt.broadcast for lhs if its shape differs from the broadcast shape.
        let lhs_out = if lhs_dims == broadcast_dims {
            lhs
        } else {
            let op: Operation<'a> = broadcast(self.module.context(), location, lhs, result_ty)
                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
                .into();
            let r = op.result(0).expect("broadcast lhs result");
            mlir_block.append_operation(op);
            r.into()
        };

        // Emit tt.broadcast for rhs if its shape differs from the broadcast shape.
        let rhs_out = if rhs_dims == broadcast_dims {
            rhs
        } else {
            let op: Operation<'a> = broadcast(self.module.context(), location, rhs, result_ty)
                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
                .into();
            let r = op.result(0).expect("broadcast rhs result");
            mlir_block.append_operation(op);
            r.into()
        };

        // Store both results as tuple fields; return None to skip ssa_values insertion.
        state.tuple_fields.insert(destination.local, vec![lhs_out, rhs_out])?;
        Ok(None)
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::broadcast_to(x, shape)` — broadcast tensor to given shape.
    //
    // `tt.broadcast` requires source dims to be either 1 (expanded) or equal to
    // the target.  When shapes are otherwise incompatible (as can happen in the
    // kitchen-sink coverage test), we return the source value unchanged so MLIR
    // verification still passes.
    // -------------------------------------------------------------------------
    pub fn codegen_broadcast_to_call<'tcx>(
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
        let src = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let shape = self.shape_from_slice_arg(&args[1].node, state, "broadcast_to")?;

        let src_tensor: RankedTensorType<'a> = src
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims =
            src_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = src_tensor.element();

        // Check that every dimension is valid for tt.broadcast:
        //   src_dim == 1 (expanding)  OR  src_dim == target_dim
        // Dynamic dims (i64::MIN) are assumed compatible.
        let broadcast_valid = src_dims.len() == shape.len()
            && src_dims
                .iter()
                .zip(shape.iter())
                .all(|(&s, &t)| s == 1 || s == t || s == i64::MIN || t == i64::MIN);

        if broadcast_valid {
            let result_ty = tensor_type(&shape, elem_ty).into();
            let op: Operation<'a> = broadcast(self.module.context(), location, src, result_ty)
                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
                .into();
            let result = op.result(0).expect("broadcast_to result");
            mlir_block.append_operation(op);
            Ok(Some(result.into()))
        } else {
            Err(MlirError::CodegenFailed {
                err: format!("broadcast_to cannot broadcast {src_dims:?} to {shape:?}"),
            })
        }
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::expand_dims(x, axis)` — insert a size-1 dimension.
    // -------------------------------------------------------------------------
    pub fn codegen_expand_dims_call<'tcx>(
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
        let src = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let axis = self
            .to_scalar_int(tcx, instance, &args[1].node)
            .map_err(|e| MlirError::CodegenFailed { err: format!("expand_dims axis: {e:?}") })?
            .to_i32();

        // When `src` is a scalar (e.g. result of tt.reduce on a 1-D tensor), emit tt.splat
        // to produce `tensor<1xElem>` rather than tt.expand_dims which requires a tensor input.
        eprintln!("[DEBUG-EXPAND-DIMS] src type: {}", src.r#type());
        if let Ok(src_tensor) = RankedTensorType::try_from(src.r#type()) {
            let src_dims =
                src_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
            let elem_ty = src_tensor.element();

            let axis_usize =
                if axis < 0 { (src_dims.len() as i32 + axis + 1) as usize } else { axis as usize };
            let mut result_dims = src_dims.clone();
            result_dims.insert(axis_usize, 1);
            let result_ty = tensor_type(&result_dims, elem_ty).into();

            let op: Operation<'a> =
                expand_dims(self.module.context(), location, src, axis, result_ty)
                    .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
                    .into();
            let result = op.result(0).expect("expand_dims result");
            mlir_block.append_operation(op);
            Ok(Some(result.into()))
        } else {
            // Scalar input — insert one unit dimension via tt.splat.
            let elem_ty = src.r#type();
            let result_ty = tensor_type(&[1], elem_ty).into();
            let op: Operation<'a> = splat(self.module.context(), location, src, result_ty)
                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
                .into();
            let result = op.result(0).expect("expand_dims(scalar) splat result");
            mlir_block.append_operation(op);
            Ok(Some(result.into()))
        }
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::permute(x, dims)` — permute dimensions.
    // Maps to `tt.trans`.
    // -------------------------------------------------------------------------
    pub fn codegen_permute_call<'tcx>(
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
        let src = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let order = self.shape_from_slice_arg(&args[1].node, state, "permute")?;
        let order_i32: Vec<i32> = order.iter().map(|&v| v as i32).collect();

        let src_tensor: RankedTensorType<'a> = src
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims =
            src_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = src_tensor.element();

        if order_i32.len() != src_dims.len() {
            return Err(MlirError::CodegenFailed {
                err: format!(
                    "permute order {order_i32:?} has {} entries for a rank-{} tensor",
                    order_i32.len(),
                    src_dims.len()
                ),
            });
        }

        // Apply permutation to get result dims.
        let result_dims: Vec<i64> = order_i32.iter().map(|&i| src_dims[i as usize]).collect();
        let result_ty = tensor_type(&result_dims, elem_ty).into();

        let op: Operation<'a> = trans(self.module.context(), location, src, &order_i32, result_ty)
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
            .into();
        let result = op.result(0).expect("permute result");
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::reshape(x, shape, can_reorder)` — reshape a tensor.
    // -------------------------------------------------------------------------
    pub fn codegen_reshape_call<'tcx>(
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
        let src = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let shape = self.shape_from_slice_arg(&args[1].node, state, "reshape")?;
        let allow_reorder = args
            .get(2)
            .and_then(|a| self.to_scalar_int(tcx, instance, &a.node).ok())
            .map(|s| s.to_u8() != 0)
            .unwrap_or(false);

        let src_tensor: RankedTensorType<'a> = src
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims =
            src_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = src_tensor.element();

        let src_numel: i64 = src_dims.iter().product();
        let dst_numel: i64 = shape.iter().product();

        // Total element counts must match (unless dims are dynamic). Returning `src` unchanged
        // instead would compile a wrong-shaped tensor that only fails later, if at all.
        if src_numel != dst_numel && src_numel > 0 && dst_numel > 0 {
            return Err(MlirError::CodegenFailed {
                err: format!(
                    "reshape from {src_dims:?} to {shape:?} changes the element count \
                     ({src_numel} vs {dst_numel})"
                ),
            });
        }

        let result_ty = tensor_type(&shape, elem_ty).into();

        let op: Operation<'a> =
            reshape(self.module.context(), location, src, result_ty, allow_reorder, false)
                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
                .into();
        let result = op.result(0).expect("reshape result");
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::trans(x, dims)` — transpose/permute dimensions.
    // Same as permute — maps to `tt.trans`.
    // -------------------------------------------------------------------------
    pub fn codegen_trans_call<'tcx>(
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
        // Delegate to permute logic.
        self.codegen_permute_call(
            tcx,
            instance,
            mir,
            _func,
            _func_name,
            args,
            _destination,
            _target,
            _unwind,
            _call_source,
            _fn_span,
            location,
            mlir_block,
            state,
        )
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::ravel(x, can_reorder)` — flatten to 1-D.
    // -------------------------------------------------------------------------
    pub fn codegen_ravel_call<'tcx>(
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
        let src = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let allow_reorder = args
            .get(1)
            .and_then(|a| self.to_scalar_int(tcx, instance, &a.node).ok())
            .map(|s| s.to_u8() != 0)
            .unwrap_or(false);

        let src_tensor: RankedTensorType<'a> = src
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims =
            src_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = src_tensor.element();

        // Compute total number of elements.
        let total: i64 = src_dims.iter().product();

        // If already 1D or total is dynamic, return src unchanged to avoid reshape issues.
        if src_dims.len() == 1 || total < 0 {
            return Ok(Some(src));
        }

        let result_ty = tensor_type(&[total], elem_ty).into();

        let op: Operation<'a> =
            reshape(self.module.context(), location, src, result_ty, allow_reorder, false)
                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
                .into();
        let result = op.result(0).expect("ravel result");
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::view(x, shape)` — reshape without reordering.
    // -------------------------------------------------------------------------
    pub fn codegen_view_call<'tcx>(
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
        let src = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let shape = self.shape_from_slice_arg(&args[1].node, state, "view")?;

        let src_tensor: RankedTensorType<'a> = src
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims =
            src_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = src_tensor.element();

        let src_numel: i64 = src_dims.iter().product();
        let dst_numel: i64 = shape.iter().product();

        if src_numel != dst_numel && src_numel > 0 && dst_numel > 0 {
            return Err(MlirError::CodegenFailed {
                err: format!(
                    "view from {src_dims:?} to {shape:?} changes the element count \
                     ({src_numel} vs {dst_numel})"
                ),
            });
        }

        let result_ty = tensor_type(&shape, elem_ty).into();

        let op: Operation<'a> =
            reshape(self.module.context(), location, src, result_ty, false, false)
                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
                .into();
        let result = op.result(0).expect("view result");
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::join(a, b)` — join along a new minor dimension.
    // -------------------------------------------------------------------------
    pub fn codegen_join_call<'tcx>(
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
        let lhs = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let rhs = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        let lhs_tensor: RankedTensorType<'a> = lhs
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims =
            lhs_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = lhs_tensor.element();

        // Result: same dims except last dim doubled (join appends a trailing dim-2, then flattens).
        // Actually tt.join appends a new trailing dimension of size 2.
        let mut result_dims = src_dims.clone();
        result_dims.push(2);
        let result_ty = tensor_type(&result_dims, elem_ty).into();

        let op: Operation<'a> = join(self.module.context(), location, lhs, rhs, result_ty)
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
            .into();
        let result = op.result(0).expect("join result");
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::interleave(a, b)` — interleave along last dimension.
    // Equivalent to join(a, b).reshape([..., 2 * last_dim]).
    // -------------------------------------------------------------------------
    pub fn codegen_interleave_call<'tcx>(
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
        let lhs = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let rhs = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        let lhs_tensor: RankedTensorType<'a> = lhs
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims =
            lhs_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = lhs_tensor.element();

        // Step 1: join(a, b) → tensor<...xN x 2 x elem>
        let mut join_dims = src_dims.clone();
        join_dims.push(2);
        let join_ty = tensor_type(&join_dims, elem_ty).into();

        let join_op: Operation<'a> = join(self.module.context(), location, lhs, rhs, join_ty)
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
            .into();
        let join_val: Value<'a, 'a> = join_op.result(0).expect("interleave join result").into();
        mlir_block.append_operation(join_op);

        // Step 2: reshape → tensor<...x 2N x elem>
        let mut result_dims = src_dims[..src_dims.len().saturating_sub(1)].to_vec();
        let last = src_dims.last().copied().unwrap_or(1);
        result_dims.push(2 * last);
        let result_ty = tensor_type(&result_dims, elem_ty).into();

        let reshape_op: Operation<'a> =
            reshape(self.module.context(), location, join_val, result_ty, false, false)
                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
                .into();
        let result = reshape_op.result(0).expect("interleave reshape result");
        mlir_block.append_operation(reshape_op);
        Ok(Some(result.into()))
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::split(x)` — split along last dimension (size must be 2).
    // Returns `(Tensor<D>, Tensor<D>)` stored in `tuple_fields`.
    // -------------------------------------------------------------------------
    pub fn codegen_split_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
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
        let src = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        let src_tensor: RankedTensorType<'a> = src
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let src_dims =
            src_tensor.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = src_tensor.element();

        let last_dim = src_dims.last().copied().unwrap_or(0);

        // tt.split requires the last dimension to be 2.
        if last_dim != 2 {
            return Err(MlirError::CodegenFailed {
                err: format!("split needs a last dimension of 2, got {src_dims:?}"),
            });
        }

        // Output type: same as input without the last dimension (which must be 2).
        let out_dims = &src_dims[..src_dims.len().saturating_sub(1)];
        let out_ty = tensor_type(out_dims, elem_ty).into();

        let op: Operation<'a> = split(self.module.context(), location, src, out_ty)
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
            .into();
        let lhs_val: Value<'a, 'a> = op.result(0).expect("split lhs").into();
        let rhs_val: Value<'a, 'a> = op.result(1).expect("split rhs").into();
        mlir_block.append_operation(op);

        // Store as tuple fields; return None to skip ssa_values insertion.
        state.tuple_fields.insert(destination.local, vec![lhs_val, rhs_val])?;
        Ok(None)
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::dot` — matrix multiply (tt.dot).
    //
    // args[0] = a: Tensor<D>
    // args[1] = b: Tensor<D>
    // args[2] = acc: Option<Tensor<O>>
    // args[3] = input_precision: InputPrecision (required, no Option — see the comment
    //           below where it's read)
    // args[4] = max_num_imprecise_acc: Option<i32>
    // -------------------------------------------------------------------------
    pub fn codegen_dot_call<'tcx>(
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
        let a = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let b = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        // tt.dot requires 2D operands with compatible shapes [M×K] × [K×N] → [M×N].
        let (a_dims, b_dims) = {
            let a_ty: Result<RankedTensorType<'a>, _> = a.r#type().try_into();
            let b_ty: Result<RankedTensorType<'a>, _> = b.r#type().try_into();
            match (a_ty, b_ty) {
                (Ok(at), Ok(bt)) => {
                    let ad = at.dims().unwrap_or_default();
                    let bd = bt.dims().unwrap_or_default();
                    (ad, bd)
                }
                _ => {
                    return Err(MlirError::CodegenFailed {
                        err: "dot operands must be ranked tensors".to_string(),
                    });
                }
            }
        };

        // Require 2D tensors with matching inner dimension.
        let dot_valid = a_dims.len() == 2
            && b_dims.len() == 2
            && (a_dims[1] == b_dims[0] || a_dims[1] == i64::MIN || b_dims[0] == i64::MIN);

        if !dot_valid {
            return Err(MlirError::CodegenFailed {
                err: format!(
                    "dot needs 2-D operands with matching inner dimensions, got {a_dims:?} and \
                     {b_dims:?}"
                ),
            });
        }

        // Accumulator: use provided acc or create a zero tensor of shape [M×N].
        let acc_opt = self.codegen_option_operand(
            tcx,
            instance,
            mir,
            &args[2].node,
            location,
            mlir_block,
            state,
        )?;

        let a_ty: RankedTensorType<'a> = a
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = a_ty.element();
        let m = a_dims[0];
        let n = b_dims[1];
        let out_ty = tensor_type(&[m, n], elem_ty).into();

        let c = match acc_opt {
            Some(v) => v,
            None => {
                // Build a zeros splat of shape [M×N].
                let zero_op: Operation<'a> = self.zero_constant(location, elem_ty);
                let zero_val = zero_op.result(0).expect("zero constant");
                mlir_block.append_operation(zero_op);

                let splat_op: Operation<'a> =
                    splat(self.module.context(), location, zero_val.into(), out_ty)
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let r = splat_op.result(0).expect("splat result").into();
                mlir_block.append_operation(splat_op);
                r
            }
        };

        // Read InputPrecision discriminant (args[3]): a required, non-Option arg on the
        // teenygrad side now — every call site must pick a precision explicitly, so there's
        // no "default" to fall back to here. Same fieldless-enum-via-scalar-int pattern as
        // DotFormat/fast_math in codegen_dot_scaled_call below.
        let precision_int = self
            .to_scalar_int(tcx, instance, &args[3].node)
            .map(|s| s.to_bits_unchecked() as i32)
            .unwrap_or(2); // IEEE, matching tt.dot's own default, only if malformed input slips through.
        let precision = match precision_int {
            0 => InputPrecision::TF32,
            1 => InputPrecision::TF32x3,
            2 => InputPrecision::IEEE,
            3 => InputPrecision::BF16x3,
            4 => InputPrecision::BF16x6,
            _ => InputPrecision::IEEE,
        };

        // max_num_imprecise_acc (Option<i32> → default 0).
        let _max_imprecise_opt = self.codegen_option_operand(
            tcx,
            instance,
            mir,
            &args[4].node,
            location,
            mlir_block,
            state,
        )?;

        let dot_op: Operation<'a> = dot(self.module.context(), location, a, b, c, precision, 0)
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
            .into();
        let result = dot_op.result(0).expect("dot result").into();
        mlir_block.append_operation(dot_op);
        Ok(Some(result))
    }

    /// The `tt.dot_scaled` element type for a `DotFormat` argument, which must be a constant.
    ///
    /// The discriminant follows `DotFormat`'s declaration order: E4M3, E5M2, E2M1x2, E2M1x4,
    /// BF16x2, Int8, UInt8. It was used to index Triton's `ScaleDotElemType` list, whose order
    /// differs, so e.g. `BF16x2` was lowered as fp4 `E2M1`.
    fn dot_scaled_elem_type<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        format: &Operand<'tcx>,
        operand: &str,
    ) -> Result<ScaleDotElemType, MlirError> {
        // DotFormat is a fieldless enum of size 1, so read its bits without `to_i32`'s size check.
        let discriminant = self
            .to_scalar_int(tcx, instance, format)
            .map(|s| s.to_bits_unchecked())
            .map_err(|_| MlirError::CodegenFailed {
                err: format!("dot_scaled: the {operand} format must be a constant `DotFormat`"),
            })?;
        let unsupported = |name: &str| MlirError::CodegenFailed {
            err: format!(
                "dot_scaled: {operand} format `DotFormat::{name}` has no `tt.dot_scaled` element \
                 type (Triton supports E4M3, E5M2, E2M1x2 and BF16x2)"
            ),
        };
        match discriminant {
            0 => Ok(ScaleDotElemType::E4M3),
            1 => Ok(ScaleDotElemType::E5M2),
            // Two fp4 values per byte along K: Triton's default `k_pack` packing.
            2 => Ok(ScaleDotElemType::E2M1),
            3 => Err(unsupported("E2M1x4")),
            4 => Ok(ScaleDotElemType::BF16),
            5 => Err(unsupported("Int8")),
            6 => Err(unsupported("UInt8")),
            other => Err(MlirError::CodegenFailed {
                err: format!("dot_scaled: unknown {operand} `DotFormat` discriminant {other}"),
            }),
        }
    }

    // -------------------------------------------------------------------------
    // `triton::Triton::dot_scaled` — scaled mixed-precision matrix multiply (tt.dot_scaled).
    //
    // args[0] = lhs: Tensor<D>
    // args[1] = lhs_scale: Tensor<S>
    // args[2] = lhs_format: DotFormat (integer discriminant)
    // args[3] = rhs: Tensor<D>
    // args[4] = rhs_scale: Tensor<S>
    // args[5] = rhs_format: DotFormat (integer discriminant)
    // args[6] = acc: Option<Tensor<O>>
    // args[7] = fast_math: bool
    // -------------------------------------------------------------------------
    pub fn codegen_dot_scaled_call<'tcx>(
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
        let lhs = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let lhs_scale = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let rhs = self.codegen_operand(
            tcx,
            instance,
            &args[3].node,
            args[3].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let rhs_scale = self.codegen_operand(
            tcx,
            instance,
            &args[4].node,
            args[4].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        // Check that lhs/rhs are 2D with compatible inner dims (lhs[-1] == rhs[-2]).
        let lhs_dims =
            lhs.r#type().try_into().ok().and_then(|t: RankedTensorType<'a>| t.dims().ok());
        let rhs_dims =
            rhs.r#type().try_into().ok().and_then(|t: RankedTensorType<'a>| t.dims().ok());

        let (lhs_dims, rhs_dims) = match (lhs_dims, rhs_dims) {
            (Some(ld), Some(rd))
                if ld.len() == 2
                    && rd.len() == 2
                    && (ld[1] == rd[0] || ld[1] == i64::MIN || rd[0] == i64::MIN) =>
            {
                (ld, rd)
            }
            (ld, rd) => {
                let describe = |d: Option<Vec<i64>>| {
                    d.map_or("a non-tensor".to_string(), |d| format!("{d:?}"))
                };
                return Err(MlirError::CodegenFailed {
                    err: format!(
                        "dot_scaled needs 2-D operands with matching inner dimensions, got {} \
                         and {}",
                        describe(ld),
                        describe(rd)
                    ),
                });
            }
        };

        let lhs_elem_type = self.dot_scaled_elem_type(tcx, instance, &args[2].node, "lhs")?;
        let rhs_elem_type = self.dot_scaled_elem_type(tcx, instance, &args[5].node, "rhs")?;

        // fast_math (args[7]): bool has size 1, use to_bits_unchecked().
        let fast_math = self
            .to_scalar_int(tcx, instance, &args[7].node)
            .map(|s| s.to_bits_unchecked() != 0)
            .unwrap_or(false);

        // Accumulator (args[6]: Option<Tensor<O>>).
        let acc_opt = self.codegen_option_operand(
            tcx,
            instance,
            mir,
            &args[6].node,
            location,
            mlir_block,
            state,
        )?;

        let lhs_ty: RankedTensorType<'a> = lhs
            .r#type()
            .try_into()
            .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = lhs_ty.element();
        let m = lhs_dims[0];
        let n = rhs_dims[1];
        let out_ty = tensor_type(&[m, n], elem_ty).into();

        let c = match acc_opt {
            Some(v) => v,
            None => {
                let zero_op: Operation<'a> = self.zero_constant(location, elem_ty);
                let zero_val = zero_op.result(0).expect("zero constant");
                mlir_block.append_operation(zero_op);

                let splat_op: Operation<'a> =
                    splat(self.module.context(), location, zero_val.into(), out_ty)
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let r = splat_op.result(0).expect("splat result").into();
                mlir_block.append_operation(splat_op);
                r
            }
        };

        let scaled_op: Operation<'a> = dot_scaled(
            self.module.context(),
            location,
            lhs,
            rhs,
            c,
            Some(lhs_scale),
            Some(rhs_scale),
            lhs_elem_type,
            rhs_elem_type,
            fast_math,
        )
        .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
        .into();
        let result = scaled_op.result(0).expect("dot_scaled result").into();
        mlir_block.append_operation(scaled_op);
        Ok(Some(result))
    }

    // =========================================================================
    // Region-building helpers (for reduce / scan combine ops)
    // =========================================================================

    /// Build a `tt.reduce` combine region whose body is a single binary MLIR op.
    ///
    /// The region has one block with two arguments of `elem_ty`.  The block
    /// applies `combine_op` (e.g. `"arith.addf"`) to those two values and
    /// terminates with `tt.reduce.return`.
    fn build_reduce_region(
        &self,
        location: Location<'a>,
        elem_ty: melior::ir::Type<'a>,
        combine_op: &str,
    ) -> Region<'a> {
        let region = Region::new();
        let block = Block::new(&[(elem_ty, location), (elem_ty, location)]);
        let lhs: Value<'_, '_> = block.argument(0).unwrap().into();
        let rhs: Value<'_, '_> = block.argument(1).unwrap().into();
        let op = OperationBuilder::new(combine_op, location)
            .add_operands(&[lhs, rhs])
            .add_results(&[elem_ty])
            .build()
            .expect("reduce combine op");
        let result: Value<'_, '_> = op.result(0).unwrap().into();
        block.append_operation(op);
        let ret: Operation<'a> = reduce_return(self.module.context(), location, &[result])
            .expect("reduce_return")
            .into();
        block.append_operation(ret);
        region.append_block(block);
        region
    }

    /// Build a `tt.scan` combine region whose body is a single binary MLIR op.
    ///
    /// Same structure as `build_reduce_region` but uses `tt.scan.return`.
    fn build_scan_region(
        &self,
        location: Location<'a>,
        elem_ty: melior::ir::Type<'a>,
        combine_op: &str,
    ) -> Region<'a> {
        let region = Region::new();
        let block = Block::new(&[(elem_ty, location), (elem_ty, location)]);
        let lhs: Value<'_, '_> = block.argument(0).unwrap().into();
        let rhs: Value<'_, '_> = block.argument(1).unwrap().into();
        let op = OperationBuilder::new(combine_op, location)
            .add_operands(&[lhs, rhs])
            .add_results(&[elem_ty])
            .build()
            .expect("scan combine op");
        let result: Value<'_, '_> = op.result(0).unwrap().into();
        block.append_operation(op);
        let ret: Operation<'a> =
            scan_return(self.module.context(), location, &[result]).expect("scan_return").into();
        block.append_operation(ret);
        region.append_block(block);
        region
    }

    /// Derive the result type for a reduction: the source tensor with its
    /// `axis` dimension removed.  For a 1-D tensor the result is the element
    /// type (a scalar).
    fn reduce_result_ty(&self, src: Value<'a, 'a>, axis: i32) -> melior::ir::Type<'a> {
        if let Ok(t) = RankedTensorType::try_from(src.r#type()) {
            let elem = t.element();
            let dims: Vec<i64> = t.dims().unwrap_or_default();
            let out_dims: Vec<i64> = dims
                .iter()
                .enumerate()
                .filter(|&(i, _)| i as i32 != axis)
                .map(|(_, &d)| d)
                .collect();
            if out_dims.is_empty() { elem } else { tensor_type(&out_dims, elem).into() }
        } else {
            src.r#type() // scalar source → scalar result
        }
    }

    /// Return the element type of a tensor value (or the type itself if scalar).
    fn elem_ty(&self, v: Value<'a, 'a>) -> melior::ir::Type<'a> {
        if let Ok(t) = RankedTensorType::try_from(v.r#type()) { t.element() } else { v.r#type() }
    }

    /// Choose the right binary combine op based on element type (float vs int).
    fn choose_float_int_op(
        elem_ty: melior::ir::Type<'_>,
        float_op: &'static str,
        int_op: &'static str,
    ) -> &'static str {
        if elem_ty.is_integer() || elem_ty.is_index() { int_op } else { float_op }
    }

    /// Like `choose_float_int_op`, but also distinguishes signed from unsigned integers
    /// (`[float_op, signed_op, unsigned_op]`). MLIR integer types are signless, so signedness
    /// comes from the Rust element type of `rust_ty` (a `Tensor<D>` or a scalar `D`); when that
    /// is not a primitive integer, the MLIR element type of `value` decides as before.
    fn choose_signed_op<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        rust_ty: Ty<'tcx>,
        value: Value<'a, 'a>,
        [float_op, signed_op, unsigned_op]: [&'static str; 3],
    ) -> &'static str {
        let rust_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, rust_ty),
        );
        match Self::rust_elem_ty(rust_ty).kind() {
            TyKind::Uint(_) => unsigned_op,
            TyKind::Int(_) => signed_op,
            _ => Self::choose_float_int_op(self.elem_ty(value), float_op, signed_op),
        }
    }

    /// Whether integer ops on `rust_ty` (a scalar, or a tensor generic over its element) need
    /// the unsigned form. MLIR integers are signless, so this has to come from the Rust type.
    /// `bool` counts as unsigned: widening it must zero-extend.
    pub(crate) fn is_unsigned<'tcx>(
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        rust_ty: Ty<'tcx>,
    ) -> bool {
        let rust_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, rust_ty),
        );
        matches!(Self::rust_elem_ty(rust_ty).kind(), TyKind::Uint(_) | TyKind::Bool)
    }

    /// The element type of a tensor type: `LlvmTensor<D>` (and any wrapper generic over its
    /// element) is followed through its first type argument down to `D`. A type without type
    /// arguments, scalar or not, is returned as is.
    fn rust_elem_ty<'tcx>(mut ty: Ty<'tcx>) -> Ty<'tcx> {
        while let TyKind::Adt(_, args) = ty.kind() {
            match args.types().next() {
                Some(inner) => ty = inner,
                None => break,
            }
        }
        ty
    }

    /// Emit `arith.constant 0.0 / 0` of `elem_ty` and splat to a tensor matching
    /// the shape of `like_val` (which may be scalar or tensor).  Used by `full`.
    fn splat_scalar_const(
        &self,
        location: Location<'a>,
        scalar_val: Value<'a, 'a>,
        result_ty: melior::ir::Type<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        if result_ty.is_tensor() {
            let splat_op: Operation<'a> =
                splat(self.module.context(), location, scalar_val, result_ty)
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into();
            let r = splat_op.result(0).unwrap().into();
            mlir_block.append_operation(splat_op);
            Ok(r)
        } else {
            Ok(scalar_val)
        }
    }

    /// Collect MLIR i64 constants from the `slice_shape` side-table for the
    /// given arg operand.  Used to reconstruct tensor shapes / strides / offsets
    /// for block-pointer ops.
    fn slice_as_i64_values<'tcx>(
        &self,
        arg: &Operand<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &CodegenState<'a, 'a>,
    ) -> Result<Vec<Value<'a, 'a>>, MlirError> {
        let local = match arg {
            Operand::Move(p) | Operand::Copy(p) => p.local,
            _ => return Err(MlirError::CodegenFailed { err: "slice arg is not a place".into() }),
        };
        // Dynamic values (runtime SSA): return them directly.
        if let Some(vals) = state.slice_dyn_values.get(&local) {
            return Ok(vals.clone());
        }
        // Static shape: create i64 constants.
        let vals = state
            .slice_shape
            .get(&local)
            .ok_or_else(|| MlirError::CodegenFailed {
                err: format!("slice_shape/slice_dyn_values not found for {:?}", local),
            })?
            .clone();
        let mut out = Vec::with_capacity(vals.len());
        for v in vals {
            let c: Operation<'a> =
                create_int_constant(self.module.context(), location, Int::I64(v as u64))
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into();
            let r = c.result(0).unwrap().into();
            mlir_block.append_operation(c);
            out.push(r);
        }
        Ok(out)
    }

    /// Same as `slice_as_i64_values` but emits i32 constants.
    fn slice_as_i32_values<'tcx>(
        &self,
        arg: &Operand<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &CodegenState<'a, 'a>,
    ) -> Result<Vec<Value<'a, 'a>>, MlirError> {
        let local = match arg {
            Operand::Move(p) | Operand::Copy(p) => p.local,
            _ => return Err(MlirError::CodegenFailed { err: "slice arg is not a place".into() }),
        };
        // Dynamic values (runtime SSA): return them directly.
        if let Some(vals) = state.slice_dyn_values.get(&local) {
            return Ok(vals.clone());
        }
        // Static shape: create i32 constants.
        let vals = state
            .slice_shape
            .get(&local)
            .ok_or_else(|| MlirError::CodegenFailed {
                err: format!("slice_shape/slice_dyn_values not found for {:?}", local),
            })?
            .clone();
        let mut out = Vec::with_capacity(vals.len());
        for v in vals {
            let c: Operation<'a> =
                create_int_constant(self.module.context(), location, Int::I32(v as u32))
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into();
            let r = c.result(0).unwrap().into();
            mlir_block.append_operation(c);
            out.push(r);
        }
        Ok(out)
    }

    /// The error for a descriptor op whose slice argument can't be resolved to per-dimension
    /// values, e.g. one chosen at run time.
    fn unresolved_slice(op: &str, arg: &str, cause: MlirError) -> MlirError {
        MlirError::CodegenFailed {
            err: format!(
                "{op}: cannot resolve `{arg}`; it must be a slice with a single definition \
                 (not chosen at run time): {cause:?}"
            ),
        }
    }

    // =========================================================================
    // Block-pointer and descriptor ops
    // =========================================================================

    /// `triton::Triton::make_block_ptr(base, shape, strides, offsets, block_shape, order)`.
    ///
    /// Emits no op: the block pointer is recorded in `state.block_ptrs` (see `BlockPtr`), the
    /// way Triton's own frontend keeps it since `tt.make_tensor_ptr` was removed.
    pub fn codegen_make_block_ptr_call<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _instance: &Instance<'tcx>,
        _mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
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
        let fail = |err: String| MlirError::CodegenFailed { err: format!("make_block_ptr: {err}") };
        if args.len() != 6 {
            return Err(fail(format!("expected 6 arguments, got {}", args.len())));
        }
        let dest = destination
            .as_local()
            .ok_or_else(|| fail(format!("unsupported destination {destination:?}")))?;
        // `base` is read directly so that a block pointer cannot itself be used as the base.
        let base = self.codegen_copy_operand(&args[0].node, state)?;
        let shape = self.slice_as_i64_extended(&args[1].node, location, mlir_block, state)?;
        let strides = self.slice_as_i64_extended(&args[2].node, location, mlir_block, state)?;
        let offsets = self.slice_as_i64_extended(&args[3].node, location, mlir_block, state)?;
        let block_shape = Self::static_slice(&args[4].node, state)
            .ok_or_else(|| fail("`block_shape` must be compile-time constants".into()))?;
        let order = Self::static_slice(&args[5].node, state)
            .ok_or_else(|| fail("`order` must be compile-time constants".into()))?;

        let rank = block_shape.len();
        if rank == 0 {
            return Err(fail("a block pointer needs at least one dimension".into()));
        }
        for (name, len) in [
            ("shape", shape.len()),
            ("strides", strides.len()),
            ("offsets", offsets.len()),
            ("order", order.len()),
        ] {
            if len != rank {
                return Err(fail(format!(
                    "`{name}` has {len} entries but `block_shape` has {rank}"
                )));
            }
        }
        if block_shape.iter().any(|&d| d <= 0) {
            return Err(fail(format!("`block_shape` entries must be positive, got {block_shape:?}")));
        }
        let mut sorted_order = order.clone();
        sorted_order.sort_unstable();
        if !sorted_order.iter().copied().eq(0..rank as i64) {
            return Err(fail(format!("`order` must be a permutation of 0..{rank}, got {order:?}")));
        }

        state
            .block_ptrs
            .insert(dest, BlockPtr { base, shape, strides, offsets, block_shape, order })?;
        Ok(None)
    }

    /// `triton::Triton::advance(ptr, offsets)`: a new block pointer whose offsets are the old
    /// ones plus `offsets`. Only the additions are emitted.
    pub fn codegen_advance_call<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _instance: &Instance<'tcx>,
        _mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
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
        let fail = |err: String| MlirError::CodegenFailed { err: format!("advance: {err}") };
        if args.len() != 2 {
            return Err(fail(format!("expected 2 arguments, got {}", args.len())));
        }
        let dest = destination
            .as_local()
            .ok_or_else(|| fail(format!("unsupported destination {destination:?}")))?;
        let src = match &args[0].node {
            Operand::Copy(p) | Operand::Move(p) => p.as_local(),
            Operand::Constant(_) | Operand::RuntimeChecks(_) => None,
        };
        let mut bp = src
            .and_then(|local| state.block_ptrs.get(&local))
            .cloned()
            .ok_or_else(|| fail("the pointer is not a `make_block_ptr` result".into()))?;
        let deltas = self.slice_as_i64_extended(&args[1].node, location, mlir_block, state)?;
        if deltas.len() != bp.offsets.len() {
            return Err(fail(format!(
                "{} offsets given for a rank-{} block pointer",
                deltas.len(),
                bp.offsets.len()
            )));
        }
        for (offset, delta) in bp.offsets.iter_mut().zip(deltas) {
            let op: Operation<'a> = melior::dialect::arith::addi(*offset, delta, location);
            *offset =
                op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?.into();
            mlir_block.append_operation(op);
        }
        state.block_ptrs.insert(dest, bp)?;
        Ok(None)
    }

    /// Reads a `Copy`/`Move` operand's SSA value; constants are not expected here.
    fn codegen_copy_operand<'tcx>(
        &self,
        operand: &Operand<'tcx>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        match operand {
            Operand::Copy(place) | Operand::Move(place) => self.codegen_copy(place, state),
            _ => Err(MlirError::CodegenFailed {
                err: format!("expected a place operand, got {operand:?}"),
            }),
        }
    }

    /// The compile-time contents of a slice argument, if they are known.
    fn static_slice<'tcx>(arg: &Operand<'tcx>, state: &CodegenState<'a, 'a>) -> Option<Vec<i64>> {
        match arg {
            Operand::Copy(p) | Operand::Move(p) => state.slice_shape.get(&p.local).cloned(),
            Operand::Constant(_) | Operand::RuntimeChecks(_) => None,
        }
    }

    /// `slice_as_i64_values`, with runtime entries sign-extended to `i64`: those keep the
    /// width of the Rust values they came from, which is `i32` in the Triton API.
    fn slice_as_i64_extended<'tcx>(
        &self,
        arg: &Operand<'tcx>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &CodegenState<'a, 'a>,
    ) -> Result<Vec<Value<'a, 'a>>, MlirError> {
        let i64_ty: melior::ir::Type<'a> = IntegerType::new(self.module.context(), 64).into();
        self.slice_as_i64_values(arg, location, mlir_block, state)?
            .into_iter()
            .map(|v| {
                if v.r#type() == i64_ty {
                    return Ok(v);
                }
                let op: Operation<'a> = create_extsi(self.module.context(), location, v, i64_ty)
                    .map_err(|e| MlirError::CreateOperation { err: e })?
                    .into();
                let r = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
                mlir_block.append_operation(op);
                Ok(r.into())
            })
            .collect()
    }

    pub fn codegen_make_tensor_descriptor_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
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
        // args: [base_ptr, &[shape_i32], &[strides_i32], &[block_shape_i32], padding]
        let [base_arg, shape_arg, strides_arg, block_shape_arg, _padding] = args else {
            return Err(MlirError::CodegenFailed {
                err: format!("make_tensor_descriptor: expected 5 arguments, got {}", args.len()),
            });
        };
        let block_shape = self
            .shape_from_slice_arg_i32(&block_shape_arg.node, state)
            .map_err(|e| Self::unresolved_slice("make_tensor_descriptor", "block_shape", e))?;
        let block_shape: Vec<i64> = block_shape.iter().map(|&v| i64::from(v)).collect();
        // Recorded for descriptor_load, whose result type is the descriptor's block.
        state.desc_block_shapes.insert(destination.local, block_shape.clone())?;

        let base = self.codegen_operand(
            tcx,
            instance,
            &base_arg.node,
            base_arg.node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        // shape: Variadic<I32> — use i32 values directly.
        let shape = self
            .slice_as_i32_values(&shape_arg.node, location, mlir_block, state)
            .map_err(|e| Self::unresolved_slice("make_tensor_descriptor", "shape", e))?;
        // strides: Variadic<I64> — get i32 values and sign-extend to i64.
        let strides_i32 = self
            .slice_as_i32_values(&strides_arg.node, location, mlir_block, state)
            .map_err(|e| Self::unresolved_slice("make_tensor_descriptor", "strides", e))?;
        let i64_ty: melior::ir::Type<'_> = IntegerType::new(self.module.context(), 64).into();
        let strides: Vec<Value<'a, 'a>> = strides_i32
            .into_iter()
            .map(|v| {
                let ext_op: Operation<'a> =
                    create_extsi(self.module.context(), location, v, i64_ty)
                        .map_err(|e| MlirError::CreateOperation { err: e })?
                        .into();
                let r = ext_op.result(0).unwrap().into();
                mlir_block.append_operation(ext_op);
                Ok(r)
            })
            .collect::<Result<Vec<_>, MlirError>>()?;

        // Build the !tt.tensordesc<BM x BN x T> result type from the block shape.
        let result_ty = {
            // Extract element type from the Rust generic type T in LlvmPointer<T>.
            let dest_ty = destination.ty(mir, tcx).ty;
            let dest_ty = instance.instantiate_mir_and_normalize_erasing_regions(
                tcx,
                TypingEnv::fully_monomorphized(),
                EarlyBinder::bind(tcx, dest_ty),
            );
            // The descriptor's element type must match the base pointer's pointee, which is
            // the same `D`, so map it the same way.
            let elem_ty = match dest_ty.kind() {
                TyKind::Adt(_, args) if !args.is_empty() => {
                    self.type_mapper.map_type(self.module.context(), &tcx, &args[0].expect_ty())
                }
                _ => {
                    return Err(MlirError::CodegenFailed {
                        err: format!(
                            "make_tensor_descriptor: expected a Pointer<D> result, got {dest_ty:?}"
                        ),
                    });
                }
            };
            let shape_str: String =
                block_shape.iter().map(|d| d.to_string()).collect::<Vec<_>>().join("x");
            // Triton's TensorDescType::verify rejects a ranked tensor element
            // type outright: "tensor descriptors must not wrap tensor types;
            // use !tt.tensordesc<shape x element-type[, layout]> instead"
            // (src/triton/lib/Dialect/Triton/IR/Types.cpp). The shape and
            // element type go in directly -- !tt.tensordesc<32x32xf32>.
            let type_str = format!("!tt.tensordesc<{shape_str}x{elem_ty}>");
            melior::ir::Type::parse(self.module.context(), &type_str).ok_or_else(|| {
                MlirError::CodegenFailed { err: format!("invalid tensor descriptor type {type_str}") }
            })?
        };

        use rustc_mlir::triton::tensor::PaddingOption;
        let op: Operation<'a> = make_tensor_descriptor(
            self.module.context(),
            location,
            base,
            &shape,
            &strides,
            PaddingOption::PadZero,
            result_ty,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    pub fn codegen_load_tensor_descriptor_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
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
        // args: [desc, &[indices_i32]]
        // Get the descriptor local so we can look up its block shape.
        let desc_local = match &args[0].node {
            Operand::Move(p) | Operand::Copy(p) if p.projection.is_empty() => Some(p.local),
            _ => None,
        };
        let desc = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let indices = self
            .slice_as_i32_values(&args[1].node, location, mlir_block, state)
            .map_err(|e| Self::unresolved_slice("load_tensor_descriptor", "offsets", e))?;
        let dest_ty = destination.ty(mir, tcx).ty;
        let dest_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, dest_ty),
        );

        // Use the descriptor's block shape (set at make_tensor_descriptor time) for the result
        // type. This gives us a statically-shaped tensor (e.g. tensor<32x32xf32>) instead of
        // the dynamically-shaped tensor<?xf32> that the type_mapper would infer from LlvmTensor<T>.
        let result_ty = if let Some(local) = desc_local {
            if let Some(block_shape) = state.desc_block_shapes.get(&local) {
                let elem_ty = self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty);
                // elem_ty is the mapped type of LlvmTensor<T>, which is tensor<?xT>.
                // Extract the element type from it (f32, i32, etc.) or use dest_ty element type.
                let scalar_ty = if let Ok(tt) = RankedTensorType::try_from(elem_ty) {
                    tt.element()
                } else {
                    elem_ty
                };
                tensor_type(block_shape, scalar_ty).into()
            } else {
                self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty)
            }
        } else {
            self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty)
        };

        let op: Operation<'a> = descriptor_load(
            self.module.context(),
            location,
            desc,
            &indices,
            result_ty,
            CacheModifier::None,
            EvictionPolicy::Normal,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    pub fn codegen_store_tensor_descriptor_call<'tcx>(
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
        // args: [desc, &[offsets_i32], src_tensor]
        let desc = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let indices = self
            .slice_as_i32_values(&args[1].node, location, mlir_block, state)
            .map_err(|e| Self::unresolved_slice("store_tensor_descriptor", "offsets", e))?;
        let src = self.codegen_operand(
            tcx,
            instance,
            &args[2].node,
            args[2].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        let op: Operation<'a> =
            descriptor_store(self.module.context(), location, desc, src, &indices)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        mlir_block.append_operation(op);
        Ok(None)
    }




    // =========================================================================
    // Control-flow / debug ops
    // =========================================================================

    /// `triton::Triton::where_` — conditional element-wise select (`arith.select`).
    pub fn codegen_where_call<'tcx>(
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
        // args: [condition, true_val, false_val]
        let cond = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let true_val = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let false_val = self.codegen_operand(
            tcx,
            instance,
            &args[2].node,
            args[2].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        // arith.select needs the condition and both values at one shape. Broadcasting the values
        // together, then the condition against them, then the values again reaches the shape
        // all three broadcast to, whichever of them is narrow.
        let (true_val, false_val) =
            self.maybe_broadcast_pair(location, true_val, false_val, mlir_block)?;
        let (cond, true_val) = self.maybe_broadcast_pair(location, cond, true_val, mlir_block)?;
        let (true_val, false_val) =
            self.maybe_broadcast_pair(location, true_val, false_val, mlir_block)?;
        let result_ty = true_val.r#type();
        let op: Operation<'a> = OperationBuilder::new("arith.select", location)
            .add_operands(&[cond, true_val, false_val])
            .add_results(&[result_ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::assume` — a no-op hint for the compiler.
    pub fn codegen_assume_call<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _instance: &Instance<'tcx>,
        _mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        _args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        _location: Location<'a>,
        _mlir_block: &BlockRef<'a, 'a>,
        _state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        Ok(None)
    }

    /// `triton::Triton::device_assert` — runtime assertion (`tt.assert`).
    pub fn codegen_device_assert_call<'tcx>(
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
        let cond = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let op: Operation<'a> = assert_op(self.module.context(), location, cond, "device_assert")
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        mlir_block.append_operation(op);
        Ok(None)
    }

    /// `triton::Triton::device_print` — debug print (`tt.print`).
    pub fn codegen_device_print_call<'tcx>(
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
        // args: [prefix_str (i64 placeholder), args...]
        // Collect printable tensor args (skip the first string arg).
        let mut print_vals: Vec<Value<'a, 'a>> = Vec::new();
        for spanned in args.iter().skip(1) {
            if let Ok(v) = self.codegen_operand(
                tcx,
                instance,
                &spanned.node,
                spanned.node.ty(mir, tcx),
                location,
                mlir_block,
                state,
            ) {
                print_vals.push(v);
            }
        }
        let is_signed: Vec<i32> = print_vals.iter().map(|_| 0i32).collect();
        let op: Operation<'a> = triton_print(
            self.module.context(),
            location,
            "device_print:",
            false,
            &print_vals,
            &is_signed,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        mlir_block.append_operation(op);
        Ok(None)
    }

    // =========================================================================
    // Tensor manipulation (flip, gather)
    // =========================================================================

    /// `triton::Triton::flip(x, dim)`: reverses `x` along `dim` (`None` is the last axis, and a
    /// negative `dim` counts from the end, as in Triton). Triton has no flip op, and its
    /// frontend's XOR-swap trick needs a power-of-two extent, so this gathers instead:
    /// `gather(x, (n - 1) - position_along_dim)`.
    pub fn codegen_flip_call<'tcx>(
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
        let fail = |err: String| MlirError::CodegenFailed { err: format!("flip: {err}") };
        if args.len() != 2 {
            return Err(fail(format!("expected 2 arguments, got {}", args.len())));
        }
        let src = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let dims = RankedTensorType::try_from(src.r#type())
            .ok()
            .and_then(|t| t.dims().ok())
            .ok_or_else(|| fail(format!("source is not a ranked tensor: {}", src.r#type())))?;
        let rank = dims.len() as i32;
        let dim = match self.const_option_i32(tcx, instance, mir, &args[1].node, state)? {
            None => rank - 1,
            Some(d) if d < 0 => d + rank,
            Some(d) => d,
        };
        let n = usize::try_from(dim)
            .ok()
            .and_then(|d| dims.get(d))
            .copied()
            .ok_or_else(|| fail(format!("dim out of range for rank {rank}")))?;
        let n = i32::try_from(n).map_err(|_| fail(format!("extent {n} does not fit in i32")))?;

        let context = self.module.context();
        let position = self.index_along_axis(location, dim, n, &dims, mlir_block)?;
        let last: Operation<'a> =
            create_int_constant(context, location, Int::I32((n - 1) as u32))
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let last_val: Value<'a, 'a> =
            last.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?.into();
        mlir_block.append_operation(last);
        let last_splat: Operation<'a> = splat(context, location, last_val, position.r#type())
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let last_tensor: Value<'a, 'a> = last_splat
            .result(0)
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?
            .into();
        mlir_block.append_operation(last_splat);
        let reversed: Operation<'a> =
            melior::dialect::arith::subi(last_tensor, position, location);
        let index: Value<'a, 'a> =
            reversed.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?.into();
        mlir_block.append_operation(reversed);

        let op: Operation<'a> = gather(context, location, src, index, dim, false, src.r#type())
            .map_err(|e| MlirError::CreateOperation { err: e })?;
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// The compile-time value of an `Option<i32>` argument. It arrives either as a MIR constant
    /// (`const Option::<i32>::Some(0_i32)`), destructured with const-eval, or as a local built
    /// by an aggregate, whose `Some` payload in `option_table` is an `arith.constant`.
    fn const_option_i32<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        operand: &Operand<'tcx>,
        state: &CodegenState<'a, 'a>,
    ) -> Result<Option<i32>, MlirError> {
        let fail = || MlirError::CodegenFailed {
            err: format!("expected a compile-time Option<i32>, got {operand:?}"),
        };
        match operand {
            Operand::Constant(c) => {
                let typing_env = TypingEnv::fully_monomorphized();
                let konst = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    typing_env,
                    EarlyBinder::bind(tcx, c.const_),
                );
                let value = konst.eval(tcx, typing_env, c.span).map_err(|_| fail())?;
                let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                    tcx,
                    typing_env,
                    EarlyBinder::bind(tcx, operand.ty(mir, tcx)),
                );
                let parts =
                    tcx.try_destructure_mir_constant_for_user_output(value, ty).ok_or_else(fail)?;
                match parts.fields {
                    [] => Ok(None),
                    [(ConstValue::Scalar(Scalar::Int(int)), _)] => Ok(Some(int.to_i32())),
                    _ => Err(fail()),
                }
            }
            Operand::Copy(p) | Operand::Move(p) => {
                match state.option_table.get(&p.local).ok_or_else(fail)? {
                    None => Ok(None),
                    Some(value) => Self::constant_i32(*value).map(Some).ok_or_else(fail),
                }
            }
            Operand::RuntimeChecks(_) => Err(fail()),
        }
    }

    /// The value of an `i32` produced by an `arith.constant`, if that is its definition.
    fn constant_i32(value: Value<'a, 'a>) -> Option<i32> {
        let result = melior::ir::operation::OperationResult::try_from(value).ok()?;
        let op = result.owner();
        if op.name().as_string_ref().as_str().ok()? != "arith.constant" {
            return None;
        }
        let attr = op.attribute("value").ok()?;
        i32::try_from(melior::ir::attribute::IntegerAttribute::try_from(attr).ok()?.value()).ok()
    }

    /// `triton::Triton::gather` — indexed gather along an axis (`tt.gather`).
    pub fn codegen_gather_call<'tcx>(
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
        // args: [src, indices, axis]
        let src = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let indices = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let axis =
            self.to_scalar_int(tcx, instance, &args[2].node).map(|s| s.to_i32()).unwrap_or(0);

        // The result has the indices' shape and the source's element type. The MIR destination,
        // a generic `Tensor<D>`, maps to a dynamically shaped tensor, which `tt.gather` rejects.
        let tensor_of = |value: Value<'a, 'a>, what: &str| {
            RankedTensorType::try_from(value.r#type()).map_err(|_| MlirError::CodegenFailed {
                err: format!("gather: the {what} must be a ranked tensor, got {}", value.r#type()),
            })
        };
        let indices_ty = tensor_of(indices, "indices")?;
        let src_ty = tensor_of(src, "source")?;
        let indices_shape = indices_ty
            .dims()
            .map_err(|e| MlirError::CodegenFailed { err: format!("gather: indices shape: {e}") })?;
        let result_ty = tensor_type(&indices_shape, src_ty.element()).into();

        let op: Operation<'a> =
            gather(self.module.context(), location, src, indices, axis, false, result_ty)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // =========================================================================
    // Unary elementwise math ops  (math.*)
    // =========================================================================
    //
    // All take a single tensor/scalar operand and return the same type.

    /// Emit a unary `math.*` or `tt.*` op.
    fn codegen_math_unary<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        mlir_op: &str,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let x = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let result_ty = x.r#type();
        let op: Operation<'a> = OperationBuilder::new(mlir_op, location)
            .add_operands(&[x])
            .add_results(&[result_ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// Emit a binary `arith.*` / `math.*` op with two equal-typed operands.
    fn codegen_binary_elementwise<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        mlir_op: &str,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let lhs = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let rhs = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let result_ty = lhs.r#type();
        let op: Operation<'a> = OperationBuilder::new(mlir_op, location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    pub fn codegen_abs_call<'tcx>(
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
        self.codegen_math_unary(tcx, instance, mir, args, "math.absf", location, mlir_block, state)
    }

    pub fn codegen_ceil_call<'tcx>(
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
        self.codegen_math_unary(tcx, instance, mir, args, "math.ceil", location, mlir_block, state)
    }

    pub fn codegen_floor_call<'tcx>(
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
        self.codegen_math_unary(tcx, instance, mir, args, "math.floor", location, mlir_block, state)
    }

    pub fn codegen_cos_call<'tcx>(
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
        self.codegen_math_unary(tcx, instance, mir, args, "math.cos", location, mlir_block, state)
    }

    pub fn codegen_sin_call<'tcx>(
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
        self.codegen_math_unary(tcx, instance, mir, args, "math.sin", location, mlir_block, state)
    }

    pub fn codegen_exp_call<'tcx>(
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
        self.codegen_math_unary(tcx, instance, mir, args, "math.exp", location, mlir_block, state)
    }

    pub fn codegen_exp2_call<'tcx>(
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
        self.codegen_math_unary(tcx, instance, mir, args, "math.exp2", location, mlir_block, state)
    }

    pub fn codegen_log_call<'tcx>(
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
        self.codegen_math_unary(tcx, instance, mir, args, "math.log", location, mlir_block, state)
    }

    pub fn codegen_log2_call<'tcx>(
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
        self.codegen_math_unary(tcx, instance, mir, args, "math.log2", location, mlir_block, state)
    }

    pub fn codegen_rsqrt_call<'tcx>(
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
        // Lower rsqrt(x) as 1.0 / sqrt_rn(x) via tt.precise_sqrt + tt.precise_divf.
        // math.rsqrt lowers to __nv_rsqrtf which requires libdevice to resolve at PTX JIT time.
        let x = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        let sqrt_op: Operation<'a> = precise_sqrt(self.module.context(), location, x)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let sqrt_val =
            sqrt_op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(sqrt_op);

        let x_ty = x.r#type();
        let elem_ty = if x_ty.is_tensor() {
            RankedTensorType::try_from(x_ty)
                .map_err(|e: melior::error::Error| MlirError::InvalidType { msg: e.to_string() })?
                .element()
        } else {
            x_ty
        };
        let one_attr = FloatAttribute::new(self.module.context(), elem_ty, 1.0);
        let one_const_op: Operation<'a> =
            melior::dialect::arith::constant(self.module.context(), one_attr.into(), location);
        let one_scalar =
            one_const_op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(one_const_op);
        let one_val = self.splat_scalar_const(location, one_scalar.into(), x_ty, mlir_block)?;

        let div_op: Operation<'a> =
            precise_divf(self.module.context(), location, one_val, sqrt_val.into())
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let result =
            div_op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(div_op);
        Ok(Some(result.into()))
    }

    pub fn codegen_sqrt_call<'tcx>(
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
        self.codegen_math_unary(tcx, instance, mir, args, "math.sqrt", location, mlir_block, state)
    }

    /// `triton::Triton::sqrt_rn` — IEEE-precise sqrt (`tt.precise_sqrt`).
    pub fn codegen_sqrt_rn_call<'tcx>(
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
        let x = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let op: Operation<'a> = precise_sqrt(self.module.context(), location, x)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    pub fn codegen_erf_call<'tcx>(
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
        self.codegen_math_unary(tcx, instance, mir, args, "math.erf", location, mlir_block, state)
    }

    /// `triton::Triton::atan` — on CUDA, uses libdevice `__nv_atanf` via `tt.extern_elementwise`,
    /// since `math.atan` has no lowering in Triton's TritonGPUToLLVM pipeline. Every other target
    /// goes through TritonCPU, which lowers `math.atan` but has no libdevice to resolve
    /// `__nv_atanf` against.
    pub fn codegen_atan_call<'tcx>(
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
        if tcx.sess.target.arch != Arch::Nvptx64 {
            return self.codegen_math_unary(tcx, instance, mir, args, "math.atan", location, mlir_block, state);
        }
        let x = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let result_ty = x.r#type();
        let op: Operation<'a> = OperationBuilder::new("tt.extern_elementwise", location)
            .add_operands(&[x])
            .add_results(&[result_ty])
            .add_attributes(&[
                (
                    melior::ir::Identifier::new(self.module.context(), "libname"),
                    melior::ir::Attribute::parse(self.module.context(), r#""libdevice""#).unwrap(),
                ),
                (
                    melior::ir::Identifier::new(self.module.context(), "libpath"),
                    melior::ir::Attribute::parse(self.module.context(), r#""""#).unwrap(),
                ),
                (
                    melior::ir::Identifier::new(self.module.context(), "symbol"),
                    melior::ir::Attribute::parse(self.module.context(), r#""__nv_atanf""#).unwrap(),
                ),
                (
                    melior::ir::Identifier::new(self.module.context(), "pure"),
                    melior::ir::Attribute::parse(self.module.context(), "true").unwrap(),
                ),
            ])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::sigmoid` — expanded as `1 / (1 + exp(-x))`.
    ///
    /// `__nv_sigmoidf` does not exist in CUDA libdevice, so we cannot use
    /// `tt.extern_elementwise`. Instead we lower inline using math ops that
    /// Triton's pipeline supports natively:
    ///   neg_x       = 0.0 - x          (codegen_sub)
    ///   exp_neg_x   = math.exp(neg_x)
    ///   one_plus    = 1.0 + exp_neg_x  (codegen_add)
    ///   result      = 1.0 / one_plus   (codegen_div)
    pub fn codegen_sigmoid_call<'tcx>(
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
        let x = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        // Helper: emit a scalar f32 constant of the given value.
        let scalar_f32 = |val: f64| -> Result<Value<'a, 'a>, MlirError> {
            let f32_ty = melior::ir::Type::float32(self.module.context());
            let attr = FloatAttribute::new(self.module.context(), f32_ty, val);
            let op: Operation<'a> =
                melior::dialect::arith::constant(self.module.context(), attr.into(), location);
            let v: Value<'a, 'a> = op.result(0).expect("scalar_f32 constant").into();
            mlir_block.append_operation(op);
            Ok(v)
        };

        // neg_x = 0.0 - x
        let zero = scalar_f32(0.0)?;
        let neg_x = self.codegen_sub(tcx, location, zero, x, mlir_block)?.expect("sigmoid: neg_x");

        // exp_neg_x = math.exp(neg_x)
        let exp_neg_x = {
            let ty = neg_x.r#type();
            let op: Operation<'a> = OperationBuilder::new("math.exp", location)
                .add_operands(&[neg_x])
                .add_results(&[ty])
                .build()
                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
            let v: Value<'a, 'a> = op.result(0).expect("sigmoid: exp_neg_x").into();
            mlir_block.append_operation(op);
            v
        };

        // one_plus = 1.0 + exp_neg_x
        let one_a = scalar_f32(1.0)?;
        let one_plus = self
            .codegen_add(tcx, location, one_a, exp_neg_x, mlir_block)?
            .expect("sigmoid: one_plus");

        // result = 1.0 / one_plus
        let one_b = scalar_f32(1.0)?;
        self.codegen_div(tcx, false, location, one_b, one_plus, mlir_block)
    }

    /// `triton::Triton::softmax` — numerically-stable softmax expanded inline.
    ///
    /// For input `tensor<Nxf32>` along the last (only) axis:
    ///   x_max  = reduce(x, max)            → f32 scalar
    ///   x_sub  = x - splat(x_max)          → tensor<Nxf32>
    ///   exp_x  = math.exp(x_sub)           → tensor<Nxf32>
    ///   sum_e  = reduce(exp_x, sum)        → f32 scalar
    ///   y      = exp_x / splat(sum_e)      → tensor<Nxf32>
    pub fn codegen_softmax_call<'tcx>(
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
        let x = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let x_ty = RankedTensorType::try_from(x.r#type()).map_err(|_| MlirError::InvalidType {
            msg: "softmax: expected ranked tensor input".into(),
        })?;
        let dims = x_ty.dims().map_err(|e| MlirError::InvalidType { msg: e.to_string() })?;
        let elem_ty = x_ty.element();
        // Triton's softmax treats `dim = None` as axis 0. Its output always has `x`'s shape, so
        // the reductions keep their axis and are broadcast back regardless of `keep_dims`.
        let axis =
            self.reduce_axis_arg(tcx, instance, mir, &args[1].node, state, dims.len())?.unwrap_or(0);
        let tensor_ty: melior::ir::Type<'a> = x.r#type();
        let binary = |name: &str, lhs: Value<'a, 'a>, rhs: Value<'a, 'a>| {
            OperationBuilder::new(name, location)
                .add_operands(&[lhs, rhs])
                .add_results(&[tensor_ty])
                .build()
                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })
        };

        let x_max = self.reduce_broadcast(x, axis, elem_ty, "arith.maximumf", location, mlir_block)?;
        let x_sub = self.append_result(binary("arith.subf", x, x_max)?, mlir_block)?;
        let exp_op = OperationBuilder::new("math.exp", location)
            .add_operands(&[x_sub])
            .add_results(&[tensor_ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let exp_x = self.append_result(exp_op, mlir_block)?;
        let sum_e = self.reduce_broadcast(exp_x, axis, elem_ty, "arith.addf", location, mlir_block)?;
        self.append_result(binary("arith.divf", exp_x, sum_e)?, mlir_block).map(Some)
    }

    /// Reduces `x` along `axis` with `combine_op` and broadcasts the result back to `x`'s
    /// shape, as `x - max(x, axis, keep_dims=True)` does in Triton.
    fn reduce_broadcast(
        &self,
        x: Value<'a, 'a>,
        axis: i32,
        elem: melior::ir::Type<'a>,
        combine_op: &str,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let context = self.module.context();
        let region = self.build_reduce_region(location, elem, combine_op);
        let op: Operation<'a> =
            reduce(context, location, &[x], &[self.reduce_result_ty(x, axis)], axis, region)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let reduced = self.append_result(op, mlir_block)?;
        let kept = self.keep_reduced_dim(location, x, axis, reduced, mlir_block)?;
        let op: Operation<'a> = broadcast(context, location, kept, x.r#type())
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        self.append_result(op, mlir_block)
    }

    // =========================================================================
    // Binary elementwise math ops
    // =========================================================================

    /// `triton::Triton::minimum` — element-wise minimum.
    pub fn codegen_minimum_call<'tcx>(
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
        let lhs = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let rhs = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        // Same constraint as codegen_maximum: operands must share the result type.
        let (lhs, rhs) = self.maybe_broadcast_pair(location, lhs, rhs, mlir_block)?;
        let op_name = self.choose_signed_op(
            tcx,
            instance,
            args[0].node.ty(mir, tcx),
            lhs,
            ["arith.minimumf", "arith.minsi", "arith.minui"],
        );
        let result_ty = lhs.r#type();
        let op: Operation<'a> = OperationBuilder::new(op_name, location)
            .add_operands(&[lhs, rhs])
            .add_results(&[result_ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::clamp` — clamped elementwise (`tt.clampf`).
    pub fn codegen_clamp_call<'tcx>(
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
        // args: [x, min_val, max_val]
        let x = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let min_val = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let max_val = self.codegen_operand(
            tcx,
            instance,
            &args[2].node,
            args[2].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let result_ty = x.r#type();
        let op: Operation<'a> = clampf(
            self.module.context(),
            location,
            x,
            min_val,
            max_val,
            PropagateNan::None,
            result_ty,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::fma` — fused multiply-add (`math.fma`).
    pub fn codegen_fma_call<'tcx>(
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
        // args: [a, b, c]  result = a*b + c
        let a = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let b = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let c = self.codegen_operand(
            tcx,
            instance,
            &args[2].node,
            args[2].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let result_ty = a.r#type();
        let op: Operation<'a> = OperationBuilder::new("math.fma", location)
            .add_operands(&[a, b, c])
            .add_results(&[result_ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::fdiv` — float division (`arith.divf`).
    pub fn codegen_fdiv_call<'tcx>(
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
        self.codegen_binary_elementwise(
            tcx,
            instance,
            mir,
            args,
            "arith.divf",
            location,
            mlir_block,
            state,
        )
    }

    /// `triton::Triton::div_rn` — IEEE-precise float division (`tt.precise_divf`).
    pub fn codegen_div_rn_call<'tcx>(
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
        let x = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let y = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let op: Operation<'a> = precise_divf(self.module.context(), location, x, y)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::cdiv` — integer ceiling division: `(a + b - 1) / b`.
    pub fn codegen_cdiv_call<'tcx>(
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
        // cdiv(a, b) = (a + b - 1) / b
        let a = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let b = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let ty = a.r#type();

        // Emit `one = arith.constant 1 : <ty>`
        let one_attr = melior::ir::Attribute::parse(self.module.context(), &format!("1 : {}", ty))
            .ok_or_else(|| MlirError::CodegenFailed { err: "cdiv: cannot parse 1 const".into() })?;
        let one_op: Operation<'a> = OperationBuilder::new("arith.constant", location)
            .add_attributes(&[(
                melior::ir::Identifier::new(self.module.context(), "value"),
                one_attr,
            )])
            .add_results(&[ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let one: Value<'a, 'a> = one_op.result(0).unwrap().into();
        mlir_block.append_operation(one_op);

        // a + b - 1
        let add_op: Operation<'a> = OperationBuilder::new("arith.addi", location)
            .add_operands(&[a, b])
            .add_results(&[ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let sum: Value<'a, 'a> = add_op.result(0).unwrap().into();
        mlir_block.append_operation(add_op);

        let sub_op: Operation<'a> = OperationBuilder::new("arith.subi", location)
            .add_operands(&[sum, one])
            .add_results(&[ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let numerator: Value<'a, 'a> = sub_op.result(0).unwrap().into();
        mlir_block.append_operation(sub_op);

        let div_op: Operation<'a> = OperationBuilder::new("arith.divsi", location)
            .add_operands(&[numerator, b])
            .add_results(&[ty])
            .build()
            .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let result: Value<'a, 'a> = div_op.result(0).unwrap().into();
        mlir_block.append_operation(div_op);
        Ok(Some(result))
    }

    /// `triton::Triton::swizzle2d(i, j, size_i, size_j, size_g) -> (new_i, new_j)`: Triton's
    /// `tl.swizzle2d` (python/triton/language/standard.py), which is plain scalar `i32`
    /// arithmetic; its `//` and `%` are C-style, so they lower to `divsi` / `remsi`.
    pub fn codegen_swizzle2d_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
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
        use melior::dialect::arith::{addi, divsi, minsi, muli, remsi, subi};

        if args.len() != 5 {
            return Err(MlirError::CodegenFailed {
                err: format!("swizzle2d: expected 5 arguments, got {}", args.len()),
            });
        }
        let mut operands = Vec::with_capacity(5);
        for arg in args {
            operands.push(self.codegen_operand(
                tcx,
                instance,
                &arg.node,
                arg.node.ty(mir, tcx),
                location,
                mlir_block,
                state,
            )?);
        }
        let [i, j, size_i, size_j, size_g] = operands[..] else {
            unreachable!("length checked above");
        };

        let emit = |op: Operation<'a>| -> Result<Value<'a, 'a>, MlirError> {
            let op = mlir_block.append_operation(op);
            Ok(op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?.into())
        };
        // Unrolled index of (i, j), and the number of elements in a group of `size_g` rows.
        let ij = emit(addi(emit(muli(i, size_j, location))?, j, location))?;
        let size_gj = emit(muli(size_g, size_j, location))?;
        // First row of (i, j)'s group; the last group may have fewer than `size_g` rows.
        let group_id = emit(divsi(ij, size_gj, location))?;
        let off_i = emit(muli(group_id, size_g, location))?;
        let group_rows = emit(minsi(emit(subi(size_i, off_i, location))?, size_g, location))?;
        // Position within the group, read column-major.
        let ij_in_group = emit(remsi(ij, size_gj, location))?;
        let new_i = emit(addi(off_i, emit(remsi(ij_in_group, group_rows, location))?, location))?;
        let new_j = emit(divsi(ij_in_group, group_rows, location))?;

        state.tuple_fields.insert(destination.local, vec![new_i, new_j])?;
        Ok(None)
    }

    // =========================================================================
    // Reductions (tt.reduce)
    // =========================================================================

    fn codegen_reduce_unary<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        combine_op_float: &'static str,
        [signed_op, unsigned_op]: [&'static str; 2],
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let src = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        if args.len() != 3 {
            return Err(MlirError::CodegenFailed {
                err: format!("reduction: expected 3 arguments, got {}", args.len()),
            });
        }
        let src_dims = Self::ranked_dims(src)?;
        let axis = self.reduce_axis_arg(tcx, instance, mir, &args[1].node, state, src_dims.len())?;
        let keep_dims = self.const_bool_arg(tcx, instance, &args[2].node);

        // Triton reduces `axis = None` over every element: flatten (order is irrelevant to a
        // reduction, so reordering is allowed) and reduce the single axis.
        let (input, axis) = match axis {
            Some(axis) => (src, axis),
            None => {
                let numel = src_dims.iter().product::<i64>();
                let flat = self.reshape_with(src, &[numel], self.elem_ty(src), true, location, mlir_block)?;
                (flat, 0)
            }
        };
        let elem = self.elem_ty(input);
        let combine_op = self.choose_signed_op(
            tcx,
            instance,
            args[0].node.ty(mir, tcx),
            input,
            [combine_op_float, signed_op, unsigned_op],
        );
        let region = self.build_reduce_region(location, elem, combine_op);
        let result_ty = self.reduce_result_ty(input, axis);
        let op: Operation<'a> =
            reduce(self.module.context(), location, &[input], &[result_ty], axis, region)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let result_val = self.append_result(op, mlir_block)?;
        if !keep_dims {
            return Ok(Some(result_val));
        }
        if input == src {
            return self.keep_reduced_dim(location, src, axis, result_val, mlir_block).map(Some);
        }
        // `None` with keep_dims: every axis is kept with extent 1.
        let kept_ty = tensor_type(&vec![1; src_dims.len()], elem).into();
        let op: Operation<'a> = splat(self.module.context(), location, result_val, kept_ty)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        self.append_result(op, mlir_block).map(Some)
    }

    /// The dimensions of a ranked tensor value.
    fn ranked_dims(value: Value<'a, 'a>) -> Result<Vec<i64>, MlirError> {
        RankedTensorType::try_from(value.r#type()).ok().and_then(|t| t.dims().ok()).ok_or_else(
            || MlirError::CodegenFailed {
                err: format!("expected a ranked tensor, got {}", value.r#type()),
            },
        )
    }

    /// A reduction's `Option<i32>` axis for a rank-`rank` source: `None` stays `None` (reduce
    /// every element), and a negative axis counts from the end, as in Triton's `_wrap_axis`.
    fn reduce_axis_arg<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        operand: &Operand<'tcx>,
        state: &CodegenState<'a, 'a>,
        rank: usize,
    ) -> Result<Option<i32>, MlirError> {
        let Some(axis) = self.const_option_i32(tcx, instance, mir, operand, state)? else {
            return Ok(None);
        };
        let rank = rank as i32;
        let wrapped = if axis < 0 { axis + rank } else { axis };
        if !(0..rank).contains(&wrapped) {
            return Err(MlirError::CodegenFailed {
                err: format!("reduction axis {axis} is out of range for a rank-{rank} tensor"),
            });
        }
        Ok(Some(wrapped))
    }

    /// Applies `keep_dims` to a `tt.reduce` result along `axis`: the result gets `src`'s shape
    /// with extent 1 at `axis`. A scalar result (1-D source) is splatted to that shape and a
    /// tensor result has the axis re-inserted with `tt.expand_dims`. Example: reduce
    /// tensor<128x4xf32> axis=1 → tensor<128xf32>; keep_dims → tensor<128x1xf32>.
    fn keep_reduced_dim(
        &self,
        location: Location<'a>,
        src: Value<'a, 'a>,
        axis: i32,
        result_val: Value<'a, 'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let mut kept_shape = Self::ranked_dims(src)?;
        let slot = usize::try_from(axis)
            .ok()
            .filter(|&a| a < kept_shape.len())
            .ok_or_else(|| MlirError::CodegenFailed {
                err: format!("keep_dims: axis {axis} out of range for {}", src.r#type()),
            })?;
        kept_shape[slot] = 1;
        let elem = if result_val.r#type().is_tensor() {
            self.elem_ty(result_val)
        } else {
            result_val.r#type()
        };
        let target_ty = tensor_type(&kept_shape, elem).into();
        let context = self.module.context();
        let op: Operation<'a> = if result_val.r#type().is_tensor() {
            expand_dims(context, location, result_val, axis, target_ty)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into()
        } else {
            splat(context, location, result_val, target_ty)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into()
        };
        self.append_result(op, mlir_block)
    }

    /// `triton::Triton::sum` — reduction sum.
    pub fn codegen_sum_call<'tcx>(
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
        self.codegen_reduce_unary(
            tcx,
            instance,
            mir,
            args,
            "arith.addf",
            ["arith.addi", "arith.addi"],
            location,
            mlir_block,
            state,
        )
    }

    /// `triton::Triton::max` — reduction max.
    pub fn codegen_max_call<'tcx>(
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
        self.codegen_reduce_unary(
            tcx,
            instance,
            mir,
            args,
            "arith.maximumf",
            ["arith.maxsi", "arith.maxui"],
            location,
            mlir_block,
            state,
        )
    }

    /// `triton::Triton::min` — reduction min.
    pub fn codegen_min_call<'tcx>(
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
        self.codegen_reduce_unary(
            tcx,
            instance,
            mir,
            args,
            "arith.minimumf",
            ["arith.minsi", "arith.minui"],
            location,
            mlir_block,
            state,
        )
    }

    /// `triton::Triton::xor_sum` — reduction XOR.
    pub fn codegen_xor_sum_call<'tcx>(
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
        self.codegen_reduce_unary(
            tcx,
            instance,
            mir,
            args,
            "arith.xori",
            ["arith.xori", "arith.xori"],
            location,
            mlir_block,
            state,
        )
    }

    // =========================================================================
    // Reductions with indices: argmax / argmin / max_with_indices / min_with_indices
    // =========================================================================

    /// `triton::Triton::max_with_indices(x, axis, tie_break_left, keep_dims)`.
    pub fn codegen_max_with_indices_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
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
        let (value, index) = self.codegen_reduce_with_indices(
            tcx,
            instance,
            mir,
            args,
            Extremum::Max,
            location,
            mlir_block,
            state,
        )?;
        state.tuple_fields.insert(destination.local, vec![value, index])?;
        Ok(None)
    }

    /// `triton::Triton::min_with_indices(x, axis, tie_break_left, keep_dims)`.
    pub fn codegen_min_with_indices_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
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
        let (value, index) = self.codegen_reduce_with_indices(
            tcx,
            instance,
            mir,
            args,
            Extremum::Min,
            location,
            mlir_block,
            state,
        )?;
        state.tuple_fields.insert(destination.local, vec![value, index])?;
        Ok(None)
    }

    /// `triton::Triton::argmax(x, axis, tie_break_left, keep_dims)`.
    pub fn codegen_argmax_call<'tcx>(
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
        self.codegen_reduce_with_indices(
            tcx,
            instance,
            mir,
            args,
            Extremum::Max,
            location,
            mlir_block,
            state,
        )
        .map(|(_, index)| Some(index))
    }

    /// `triton::Triton::argmin(x, axis, tie_break_left, keep_dims)`.
    pub fn codegen_argmin_call<'tcx>(
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
        self.codegen_reduce_with_indices(
            tcx,
            instance,
            mir,
            args,
            Extremum::Min,
            location,
            mlir_block,
            state,
        )
        .map(|(_, index)| Some(index))
    }

    /// Shared lowering of the four `(x, axis, tie_break_left, keep_dims)` index reductions,
    /// following Triton's `_reduce_with_indices`: reduce `(x, arange)` along `axis` with a
    /// combine that keeps the extreme value and its index, returning `(value, index)`.
    fn codegen_reduce_with_indices<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        extremum: Extremum,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(Value<'a, 'a>, Value<'a, 'a>), MlirError> {
        let fail = |err: String| MlirError::CodegenFailed {
            err: format!("{extremum:?} reduction with indices: {err}"),
        };
        if args.len() != 4 {
            return Err(fail(format!("expected 4 arguments, got {}", args.len())));
        }
        let src = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let axis = self
            .to_scalar_int(tcx, instance, &args[1].node)
            .map(|s| s.to_i32())
            .map_err(|_| fail("`axis` must be a compile-time constant".into()))?;
        let tie_break_left = self.const_bool_arg(tcx, instance, &args[2].node);
        let keep_dims = self.const_bool_arg(tcx, instance, &args[3].node);

        let dims = RankedTensorType::try_from(src.r#type())
            .ok()
            .and_then(|t| t.dims().ok())
            .ok_or_else(|| fail(format!("source is not a ranked tensor: {}", src.r#type())))?;
        let axis_len = usize::try_from(axis)
            .ok()
            .and_then(|a| dims.get(a))
            .copied()
            .ok_or_else(|| fail(format!("axis {axis} out of range for rank {}", dims.len())))?;
        let axis_len = i32::try_from(axis_len)
            .map_err(|_| fail(format!("axis length {axis_len} does not fit in i32")))?;
        let index = self.index_along_axis(location, axis, axis_len, &dims, mlir_block)?;

        let elem = self.elem_ty(src);
        let i32_ty: melior::ir::Type<'a> = IntegerType::new(self.module.context(), 32).into();
        let rust_elem = Self::rust_elem_ty(instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, args[0].node.ty(mir, tcx)),
        ));
        let compare = ValueCompare::for_elem(rust_elem.kind(), elem, extremum);

        let value_ty = self.reduce_result_ty(src, axis);
        let index_ty = self.reduce_result_ty(index, axis);
        let op: Operation<'a> = reduce(
            self.module.context(),
            location,
            &[src, index],
            &[value_ty, index_ty],
            axis,
            Region::new(),
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        self.build_index_combine(&op, elem, i32_ty, compare, tie_break_left, location)?;
        let value: Value<'a, 'a> =
            op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?.into();
        let index: Value<'a, 'a> =
            op.result(1).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?.into();
        mlir_block.append_operation(op);

        if keep_dims {
            let value = self.keep_reduced_dim(location, src, axis, value, mlir_block)?;
            let index = self.keep_reduced_dim(location, src, axis, index, mlir_block)?;
            return Ok((value, index));
        }
        Ok((value, index))
    }

    /// An `i32` tensor of `dims`' shape holding each element's position along `axis`:
    /// `arange(0, axis_len)` expanded over the other axes and broadcast.
    fn index_along_axis(
        &self,
        location: Location<'a>,
        axis: i32,
        axis_len: i32,
        dims: &[i64],
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let context = self.module.context();
        let i32_ty: melior::ir::Type<'a> = IntegerType::new(context, 32).into();
        let range: Operation<'a> = make_range(context, location, 0, axis_len)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let mut index: Value<'a, 'a> =
            range.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?.into();
        mlir_block.append_operation(range);
        if dims.len() == 1 {
            return Ok(index);
        }

        // Insert a unit dimension at every position except `axis`, in increasing order, so
        // that each insertion index is already its final position.
        let mut shape = vec![i64::from(axis_len)];
        for d in 0..dims.len() {
            if d as i32 == axis {
                continue;
            }
            shape.insert(d, 1);
            let ty = tensor_type(&shape, i32_ty).into();
            let op: Operation<'a> = expand_dims(context, location, index, d as i32, ty)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
            index = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?.into();
            mlir_block.append_operation(op);
        }
        let op: Operation<'a> =
            broadcast(context, location, index, tensor_type(dims, i32_ty).into())
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let index = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(index.into())
    }

    /// Fills the combine region of an index reduction. Its block takes `(v1, i1, v2, i2)` and
    /// keeps the first pair when `v1` is strictly better than `v2`, or, with `tie_break_left`,
    /// when they are equal and `i1 < i2` (Triton's `_argmax_combine` / `_argmin_combine`).
    fn build_index_combine(
        &self,
        op: &Operation<'a>,
        elem: melior::ir::Type<'a>,
        i32_ty: melior::ir::Type<'a>,
        compare: ValueCompare,
        tie_break_left: bool,
        location: Location<'a>,
    ) -> Result<(), MlirError> {
        use melior::dialect::arith;

        let context = self.module.context();
        let to_err = |e: melior::Error| MlirError::CodegenFailed { err: e.to_string() };
        let region = op.region(0).map_err(to_err)?;
        let block = region.append_block(Block::new(&[
            (elem, location),
            (i32_ty, location),
            (elem, location),
            (i32_ty, location),
        ]));
        let arg = |i| block.argument(i).map(Value::from).map_err(to_err);
        let (v1, i1, v2, i2) = (arg(0)?, arg(1)?, arg(2)?, arg(3)?);

        let append = |op: Operation<'a>| -> Result<Value<'a, 'a>, MlirError> {
            let op = block.append_operation(op);
            Ok(op.result(0).map_err(to_err)?.into())
        };
        let mut keep_first = append(compare.better(context, v1, v2, location))?;
        if tie_break_left {
            let equal = append(compare.equal(context, v1, v2, location))?;
            let lower = append(arith::cmpi(context, arith::CmpiPredicate::Slt, i1, i2, location))?;
            let tie = append(arith::andi(equal, lower, location))?;
            keep_first = append(arith::ori(keep_first, tie, location))?;
        }
        let value = append(arith::select(keep_first, v1, v2, location))?;
        let index = append(arith::select(keep_first, i1, i2, location))?;
        let ret: Operation<'a> = reduce_return(context, location, &[value, index])
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        block.append_operation(ret);
        Ok(())
    }

    // =========================================================================
    // Scans (tt.scan)
    // =========================================================================

    fn codegen_scan_unary<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        combine_op_float: &'static str,
        combine_op_int: &'static str,
        reverse: bool,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let src = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let axis = if args.len() > 1 {
            self.to_scalar_int(tcx, instance, &args[1].node).map(|s| s.to_i32()).unwrap_or(0)
        } else {
            0
        };
        let elem = self.elem_ty(src);
        let combine_op = Self::choose_float_int_op(elem, combine_op_float, combine_op_int);
        let region = self.build_scan_region(location, elem, combine_op);
        let result_ty = src.r#type();
        let op: Operation<'a> =
            scan(self.module.context(), location, &[src], &[result_ty], axis, reverse, region)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::cumsum` — prefix sum scan.
    pub fn codegen_cumsum_call<'tcx>(
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
        self.codegen_scan_unary(
            tcx,
            instance,
            mir,
            args,
            "arith.addf",
            "arith.addi",
            false,
            location,
            mlir_block,
            state,
        )
    }

    /// `triton::Triton::cumprod` — prefix product scan.
    pub fn codegen_cumprod_call<'tcx>(
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
        self.codegen_scan_unary(
            tcx,
            instance,
            mir,
            args,
            "arith.mulf",
            "arith.muli",
            false,
            location,
            mlir_block,
            state,
        )
    }

    /// `triton::Triton::sort(x, dim, descending)`: Triton's `tl.sort` (`sort_impl` in
    /// python/triton/language/standard.py), a bitonic sort on `x` viewed as a `[2; log2(numel)]`
    /// hypercube. Like Triton, only the last dimension is supported and every extent must be a
    /// power of two.
    pub fn codegen_sort_call<'tcx>(
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
        let fail = |err: String| MlirError::CodegenFailed { err: format!("sort: {err}") };
        if args.len() != 3 {
            return Err(fail(format!("expected 3 arguments, got {}", args.len())));
        }
        let src = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let dims = RankedTensorType::try_from(src.r#type())
            .ok()
            .and_then(|t| t.dims().ok())
            .ok_or_else(|| fail(format!("source is not a ranked tensor: {}", src.r#type())))?;
        let rank = dims.len() as i32;
        let dim = match self.const_option_i32(tcx, instance, mir, &args[1].node, state)? {
            None => rank - 1,
            Some(d) if d < 0 => d + rank,
            Some(d) => d,
        };
        if dim != rank - 1 {
            return Err(fail(format!("only the last dimension can be sorted, got dim {dim}")));
        }
        let descending = self.const_bool_arg(tcx, instance, &args[2].node);
        if let Some(d) = dims.iter().find(|&&d| d <= 0 || (d & (d - 1)) != 0) {
            return Err(fail(format!("every extent must be a power of two, got {d} in {dims:?}")));
        }
        let numel: i64 = dims.iter().product();
        if numel == 1 {
            return Ok(Some(src));
        }
        let n_dims = numel.trailing_zeros();
        let log_n = dims[dims.len() - 1].trailing_zeros();

        let elem = self.elem_ty(src);
        let rust_elem = Self::rust_elem_ty(instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, args[0].node.ty(mir, tcx)),
        ));
        let sorter = Bitonic {
            n_dims,
            elem,
            int_elem: self.same_width_int(elem).ok_or_else(|| fail(format!("unsupported element type {elem}")))?,
            compare: ValueCompare::for_elem(rust_elem.kind(), elem, Extremum::Max),
            location,
        };

        let mut h = self.reshape_to(src, &vec![2; n_dims as usize], elem, location, mlir_block)?;
        for stage in 1..=log_n {
            let order = if stage < log_n {
                BitonicOrder::Alternating
            } else {
                BitonicOrder::Fixed { descending }
            };
            h = self.bitonic_merge(&sorter, h, stage, order, mlir_block)?;
        }
        self.reshape_to(h, &dims, elem, location, mlir_block).map(Some)
    }

    /// Triton's `_bitonic_merge_hypercube`: `stage` compare-and-swap rounds, innermost last.
    fn bitonic_merge(
        &self,
        sorter: &Bitonic<'a>,
        mut h: Value<'a, 'a>,
        stage: u32,
        order: BitonicOrder,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let flip = match order {
            BitonicOrder::Alternating => Flip::Indicator(stage),
            BitonicOrder::Fixed { descending: false } => Flip::Zero,
            BitonicOrder::Fixed { descending: true } => Flip::One,
        };
        for round in 0..stage {
            h = self.compare_and_swap(sorter, h, flip, stage - 1 - round, mlir_block)?;
        }
        Ok(h)
    }

    /// Triton's `_compare_and_swap` on the `i`-th innermost hypercube axis: each element's
    /// partner `y` along that axis is `x ^ xor_sum(x)` over the axis (on the bits), and the pair
    /// is put in order with `where((x > y) != (flip ^ is_right), y, x)`.
    fn compare_and_swap(
        &self,
        sorter: &Bitonic<'a>,
        x: Value<'a, 'a>,
        flip: Flip,
        i: u32,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        use melior::dialect::arith::{CmpiPredicate, cmpi, select, xori};

        let Bitonic { n_dims, elem, int_elem, compare, location } = *sorter;
        let context = self.module.context();
        let shape = vec![2; n_dims as usize];
        let int_ty: melior::ir::Type<'a> = tensor_type(&shape, int_elem).into();
        let emit = |op: Operation<'a>| self.append_result(op, mlir_block);
        let to_err = |e| MlirError::CreateOperation { err: e };

        let ix = if elem == int_elem { x } else { emit(bitcast(context, location, x, int_ty).map_err(to_err)?)? };
        let axis = (n_dims - 1 - i) as i32;
        let reduced: Operation<'a> = reduce(
            context,
            location,
            &[ix],
            &[self.reduce_result_ty(ix, axis)],
            axis,
            self.build_reduce_region(location, int_elem, "arith.xori"),
        )
        .map_err(to_err)?
        .into();
        let mut kept_shape = shape.clone();
        kept_shape[axis as usize] = 1;
        let kept_ty: melior::ir::Type<'a> = tensor_type(&kept_shape, int_elem).into();
        let reduced = emit(reduced)?;
        let kept = if n_dims == 1 {
            emit(splat(context, location, reduced, kept_ty).map_err(to_err)?.into())?
        } else {
            emit(expand_dims(context, location, reduced, axis, kept_ty).map_err(to_err)?.into())?
        };
        let partner_bits = emit(xori(ix, emit(broadcast(context, location, kept, int_ty).map_err(to_err)?.into())?, location))?;
        let y = if elem == int_elem {
            partner_bits
        } else {
            emit(bitcast(context, location, partner_bits, x.r#type()).map_err(to_err)?)?
        };

        let is_right = self.hypercube_indicator(n_dims, i, location, mlir_block)?;
        let swap_right = match flip {
            Flip::Zero => is_right,
            Flip::One => emit(xori(is_right, self.splat_i32(1, is_right.r#type(), location, mlir_block)?, location))?,
            Flip::Indicator(stage) => {
                emit(xori(self.hypercube_indicator(n_dims, stage, location, mlir_block)?, is_right, location))?
            }
        };
        let zero = self.splat_i32(0, swap_right.r#type(), location, mlir_block)?;
        let swap_right = emit(cmpi(context, CmpiPredicate::Ne, swap_right, zero, location))?;
        let greater = emit(compare.better(context, x, y, location))?;
        let take_partner = emit(xori(greater, swap_right, location))?;
        emit(select(take_partner, y, x, location))
    }

    /// Triton's `_indicator(n_dims, j)` broadcast to the full hypercube: an `i32` tensor of
    /// shape `[2; n_dims]` holding each element's coordinate (0 or 1) on the `j`-th innermost
    /// axis.
    fn hypercube_indicator(
        &self,
        n_dims: u32,
        j: u32,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let context = self.module.context();
        let i32_ty: melior::ir::Type<'a> = IntegerType::new(context, 32).into();
        let range: Operation<'a> = make_range(context, location, 0, 2)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let range = self.append_result(range, mlir_block)?;
        let mut shape = vec![1; n_dims as usize];
        shape[(n_dims - 1 - j) as usize] = 2;
        let shaped = self.reshape_to(range, &shape, i32_ty, location, mlir_block)?;
        let full: melior::ir::Type<'a> = tensor_type(&vec![2; n_dims as usize], i32_ty).into();
        let op: Operation<'a> = broadcast(context, location, shaped, full)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        self.append_result(op, mlir_block)
    }

    /// `tt.reshape` of `value` to `shape` with element type `elem`; a no-op if the shape
    /// already matches.
    fn reshape_to(
        &self,
        value: Value<'a, 'a>,
        shape: &[i64],
        elem: melior::ir::Type<'a>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        self.reshape_with(value, shape, elem, false, location, mlir_block)
    }

    /// `reshape_to` with an explicit `allow_reorder`, which lets Triton pick any element
    /// order (fine when the consumer, such as a reduction, does not depend on it).
    fn reshape_with(
        &self,
        value: Value<'a, 'a>,
        shape: &[i64],
        elem: melior::ir::Type<'a>,
        allow_reorder: bool,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let ty: melior::ir::Type<'a> = tensor_type(shape, elem).into();
        if value.r#type() == ty {
            return Ok(value);
        }
        let op = reshape(self.module.context(), location, value, ty, allow_reorder, false)
            .map_err(|e| MlirError::CreateOperation { err: e })?;
        self.append_result(op, mlir_block)
    }

    /// The signless integer type with `elem`'s bit width (`elem` itself for integers), used to
    /// XOR element bits as Triton's `_compare_and_swap` does.
    fn same_width_int(&self, elem: melior::ir::Type<'a>) -> Option<melior::ir::Type<'a>> {
        if elem.is_integer() {
            return Some(elem);
        }
        let width = if elem.is_f16() || elem.is_bf16() {
            16
        } else if elem.is_f32() {
            32
        } else if elem.is_f64() {
            64
        } else {
            return None;
        };
        Some(IntegerType::new(self.module.context(), width).into())
    }

    /// `triton::Triton::histogram` — `tt.histogram`.
    pub fn codegen_histogram_call<'tcx>(
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
        // args: [x, num_bins, Option<mask>]
        let src = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        // The mask is the third argument; reading the second passed `num_bins` as the mask.
        let mask = self.codegen_option_operand(
            tcx,
            instance,
            mir,
            &args[2].node,
            location,
            mlir_block,
            state,
        )?;

        // One count per bin: `tensor<num_bins x i32>`. The MIR destination, a generic
        // `I32Tensor`, maps to a dynamically shaped tensor, which `tt.histogram` rejects.
        let num_bins = self.to_scalar_int(tcx, instance, &args[1].node).map_err(|_| {
            MlirError::CodegenFailed { err: "histogram: `num_bins` must be a constant".to_string() }
        })?;
        let i32_ty = IntegerType::new(self.module.context(), 32).into();
        let result_ty = tensor_type(&[i64::from(num_bins.to_i32())], i32_ty).into();

        let op: Operation<'a> = histogram(self.module.context(), location, src, mask, result_ty)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // =========================================================================
    // User-combine reduce / scan (tt.reduce / tt.scan with a combine fn)
    // =========================================================================

    /// `triton::Triton::reduce(x, axis, combine_fn, keep_dims)`.
    pub fn codegen_reduce_call<'tcx>(
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
        let (src, axis, combine) =
            self.combine_call_args(tcx, instance, mir, args, location, mlir_block, state)?;
        let keep_dims = self.const_bool_arg(tcx, instance, &args[3].node);
        let result_ty = self.reduce_result_ty(src, axis);
        let op: Operation<'a> =
            reduce(self.module.context(), location, &[src], &[result_ty], axis, Region::new())
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
        self.codegen_combine_region(tcx, combine, &op, src, location, |ctx, loc, value| {
            reduce_return(ctx, loc, &[value]).map(Into::into)
        })?;
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        let result_val: Value<'a, 'a> = result.into();
        if keep_dims {
            return self.keep_reduced_dim(location, src, axis, result_val, mlir_block).map(Some);
        }
        Ok(Some(result_val))
    }

    /// `triton::Triton::associative_scan(x, axis, combine_fn, reverse)`.
    pub fn codegen_associative_scan_call<'tcx>(
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
        let (src, axis, combine) =
            self.combine_call_args(tcx, instance, mir, args, location, mlir_block, state)?;
        let reverse = self.const_bool_arg(tcx, instance, &args[3].node);
        let op: Operation<'a> = scan(
            self.module.context(),
            location,
            &[src],
            &[src.r#type()],
            axis,
            reverse,
            Region::new(),
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        self.codegen_combine_region(tcx, combine, &op, src, location, |ctx, loc, value| {
            scan_return(ctx, loc, &[value]).map(Into::into)
        })?;
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// Decodes the `(x, axis, combine_fn, _)` arguments shared by `reduce` and
    /// `associative_scan`: the source tensor value, the constant axis, and the resolved
    /// combine fn instance.
    fn combine_call_args<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<(Value<'a, 'a>, i32, Instance<'tcx>), MlirError> {
        if args.len() != 4 {
            return Err(MlirError::CodegenFailed {
                err: format!("reduce/associative_scan: expected 4 arguments, got {}", args.len()),
            });
        }
        let src = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let axis = self.to_scalar_int(tcx, instance, &args[1].node).map(|s| s.to_i32()).map_err(
            |_| MlirError::CodegenFailed {
                err: "reduce/associative_scan: `axis` must be a compile-time constant".into(),
            },
        )?;
        let combine = self.resolve_fn_ptr_instance(tcx, instance, mir, &args[2].node)?;
        Ok((src, axis, combine))
    }

    /// Reads a constant `bool` argument; `bool` is one byte, so `to_bits_unchecked` is used
    /// rather than `to_i32`, which would panic on the size mismatch.
    fn const_bool_arg<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        operand: &Operand<'tcx>,
    ) -> bool {
        self.to_scalar_int(tcx, instance, operand).is_ok_and(|s| s.to_bits_unchecked() != 0)
    }

    /// Finds the fn item a combine-fn operand refers to. Triton has no runtime function
    /// pointers, so the operand must trace back, through `ReifyFnPointer` casts and plain
    /// copies within `mir`, to a constant fn item; a fn pointer received as a parameter
    /// cannot be resolved and is reported as an error.
    fn resolve_fn_ptr_instance<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        operand: &Operand<'tcx>,
    ) -> Result<Instance<'tcx>, MlirError> {
        let unresolved = |why: &str| MlirError::CodegenFailed {
            err: format!(
                "reduce/associative_scan: cannot resolve the combine function ({why}); it must \
                 be a fn item named at the call site, not a runtime fn pointer"
            ),
        };
        let mut operand = operand;
        // Each hop follows one assignment, so the chain cannot be longer than the body.
        for _ in 0..=mir.local_decls.len() {
            let place = match operand {
                Operand::Constant(c) => {
                    let ty = instance.instantiate_mir_and_normalize_erasing_regions(
                        tcx,
                        TypingEnv::fully_monomorphized(),
                        EarlyBinder::bind(tcx, c.const_.ty()),
                    );
                    let TyKind::FnDef(def_id, fn_args) = *ty.kind() else {
                        return Err(unresolved(&format!("constant of type {ty:?}")));
                    };
                    let fn_args = fn_args
                        .no_bound_vars()
                        .ok_or_else(|| unresolved(&format!("{def_id:?} has bound generic args")))?;
                    return Instance::resolve_for_fn_ptr(
                        tcx,
                        TypingEnv::fully_monomorphized(),
                        def_id,
                        fn_args,
                    )
                    .ok_or_else(|| unresolved(&format!("{def_id:?} has no instance")));
                }
                Operand::Copy(place) | Operand::Move(place) if place.projection.is_empty() => {
                    place
                }
                _ => return Err(unresolved("projected operand")),
            };
            operand = mir
                .basic_blocks
                .iter()
                .flat_map(|bb| bb.statements.iter())
                .find_map(|stmt| match &stmt.kind {
                    StatementKind::Assign(assign) if assign.0.as_local() == Some(place.local) => {
                        match &assign.1 {
                            Rvalue::Cast(
                                CastKind::PointerCoercion(PointerCoercion::ReifyFnPointer(_), _),
                                op,
                                _,
                            )
                            | Rvalue::Use(op, _) => Some(op),
                            _ => None,
                        }
                    }
                    _ => None,
                })
                .ok_or_else(|| unresolved(&format!("{:?} is not assigned a fn item", place.local)))?;
        }
        Err(unresolved("assignment cycle"))
    }

    /// Fills `op`'s (empty) combine region by generating `combine`'s body inline, with its two
    /// `Tensor<D>` parameters bound to the region's scalar block arguments. The handlers used
    /// for kernel code are type-driven by their operands, so they emit the scalar form of each
    /// op here. The body must be straight-line: calls, gotos and drops are followed to the
    /// `Return`, whose value `make_return` wraps in the region terminator.
    fn codegen_combine_region<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        combine: Instance<'tcx>,
        op: &Operation<'a>,
        src: Value<'a, 'a>,
        location: Location<'a>,
        make_return: impl Fn(
            &'a melior::Context,
            Location<'a>,
            Value<'a, 'a>,
        ) -> Result<Operation<'a>, rustc_mlir::errors::Error>,
    ) -> Result<(), MlirError> {
        let combine_name = tcx.def_path_str(combine.def_id());
        let unsupported = |what: String| MlirError::CodegenFailed {
            err: format!("reduce/associative_scan combine fn `{combine_name}`: {what}"),
        };
        let mir = tcx.instance_mir(combine.def);
        if mir.arg_count != 2 {
            return Err(unsupported(format!("expected 2 parameters, found {}", mir.arg_count)));
        }

        let elem = self.elem_ty(src);
        let region = op.region(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        let block = region.append_block(Block::new(&[(elem, location), (elem, location)]));

        let mut state = CodegenState::new();
        for i in 0..2 {
            let arg = block.argument(i).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
            state.ssa_values.insert(Local::from_usize(i + 1), arg.into());
        }

        let mut visited = HashSet::new();
        let mut bb = START_BLOCK;
        loop {
            if !visited.insert(bb) {
                return Err(unsupported("loops are not supported in a combine body".into()));
            }
            let data = &mir.basic_blocks[bb];
            for stmt in &data.statements {
                self.codegen_statement(tcx, &combine, mir, stmt, &block, &mut state)?;
            }
            let terminator = data.terminator();
            bb = match &terminator.kind {
                TerminatorKind::Call {
                    func,
                    args,
                    destination,
                    target: target @ Some(next),
                    unwind,
                    call_source,
                    fn_span,
                } => {
                    let call_loc = span_to_location(self.module.context(), tcx, *fn_span);
                    self.dispatch_call(
                        tcx,
                        &combine,
                        mir,
                        func,
                        args,
                        destination,
                        target,
                        unwind,
                        call_source,
                        fn_span,
                        call_loc,
                        &block,
                        &mut state,
                    )?;
                    *next
                }
                TerminatorKind::Goto { target } | TerminatorKind::Drop { target, .. } => *target,
                TerminatorKind::Return => {
                    let value = *state
                        .ssa_values
                        .get(&RETURN_PLACE)
                        .ok_or_else(|| unsupported("return value was not generated".into()))?;
                    let ret = make_return(self.module.context(), location, value)
                        .map_err(|e| MlirError::CreateOperation { err: e })?;
                    block.append_operation(ret);
                    return Ok(());
                }
                kind => {
                    return Err(unsupported(format!(
                        "unsupported terminator {:?} in a combine body",
                        kind.name()
                    )));
                }
            };
        }
    }

    // =========================================================================
    // Atomic ops (tt.atomic_rmw / tt.atomic_cas)
    // =========================================================================

    fn codegen_atomic_rmw_impl<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        rmw_op: RmwOp,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        // args: [ptr, val, Option<mask>]
        let ptr = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let val = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let mask = if args.len() > 2 {
            self.codegen_option_operand(
                tcx,
                instance,
                mir,
                &args[2].node,
                location,
                mlir_block,
                state,
            )?
        } else {
            None
        };
        let result_ty = val.r#type();
        let op: Operation<'a> = atomic_rmw(
            self.module.context(),
            location,
            ptr,
            val,
            mask,
            result_ty,
            rmw_op,
            MemSemantic::Relaxed,
            MemSyncScope::Gpu,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    pub fn codegen_atomic_add_call<'tcx>(
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
        // Use Fadd for floats; Add for ints.  Default to Fadd if type can't be inferred.
        let rmw = match self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        ) {
            Ok(v) => {
                let elem = self.elem_ty(v);
                if elem.is_integer() || elem.is_index() { RmwOp::Add } else { RmwOp::Fadd }
            }
            Err(_) => RmwOp::Fadd,
        };
        self.codegen_atomic_rmw_impl(tcx, instance, mir, args, rmw, location, mlir_block, state)
    }

    pub fn codegen_atomic_max_call<'tcx>(
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
        self.codegen_atomic_rmw_impl(
            tcx,
            instance,
            mir,
            args,
            RmwOp::Max,
            location,
            mlir_block,
            state,
        )
    }

    pub fn codegen_atomic_min_call<'tcx>(
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
        self.codegen_atomic_rmw_impl(
            tcx,
            instance,
            mir,
            args,
            RmwOp::Min,
            location,
            mlir_block,
            state,
        )
    }

    pub fn codegen_atomic_xchg_call<'tcx>(
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
        self.codegen_atomic_rmw_impl(
            tcx,
            instance,
            mir,
            args,
            RmwOp::Xchg,
            location,
            mlir_block,
            state,
        )
    }

    pub fn codegen_atomic_and_call<'tcx>(
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
        self.codegen_atomic_rmw_impl(
            tcx,
            instance,
            mir,
            args,
            RmwOp::And,
            location,
            mlir_block,
            state,
        )
    }

    pub fn codegen_atomic_or_call<'tcx>(
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
        self.codegen_atomic_rmw_impl(
            tcx,
            instance,
            mir,
            args,
            RmwOp::Or,
            location,
            mlir_block,
            state,
        )
    }

    pub fn codegen_atomic_xor_call<'tcx>(
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
        self.codegen_atomic_rmw_impl(
            tcx,
            instance,
            mir,
            args,
            RmwOp::Xor,
            location,
            mlir_block,
            state,
        )
    }

    /// `triton::Triton::atomic_cas` — `tt.atomic_cas`.
    pub fn codegen_atomic_cas_call<'tcx>(
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
        // args: [ptr, cmp, val]
        let ptr = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let cmp = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let val = self.codegen_operand(
            tcx,
            instance,
            &args[2].node,
            args[2].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let result_ty = val.r#type();
        let op: Operation<'a> = atomic_cas(
            self.module.context(),
            location,
            ptr,
            cmp,
            val,
            result_ty,
            MemSemantic::Relaxed,
            MemSyncScope::Gpu,
        )
        .map_err(|e| MlirError::CreateOperation { err: e })?
        .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // =========================================================================
    // Miscellaneous ops
    // =========================================================================

    /// `triton::Triton::umulhi` — unsigned integer multiply high (`tt.mulhiui`).
    pub fn codegen_umulhi_call<'tcx>(
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
        let x = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let y = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        let op: Operation<'a> = mulhiui(self.module.context(), location, x, y)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    // =========================================================================
    // Random numbers (Philox), following python/triton/language/random.py
    // =========================================================================

    /// `triton::Triton::randint4x(seed, offsets, n_rounds)`: the four Philox output streams.
    pub fn codegen_randint4x_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
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
        let streams = self.philox_streams(tcx, instance, mir, args, location, mlir_block, state)?;
        state.tuple_fields.insert(destination.local, streams.to_vec())?;
        Ok(None)
    }

    /// `triton::Triton::randint(seed, offsets, n_rounds)`: the first Philox stream.
    pub fn codegen_randint_call<'tcx>(
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
        let [c0, ..] = self.philox_streams(tcx, instance, mir, args, location, mlir_block, state)?;
        Ok(Some(c0))
    }

    /// `triton::Triton::rand(seed, offsets, n_rounds)`: uniform `f32` in [0, 1).
    pub fn codegen_rand_call<'tcx>(
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
        let [c0, ..] = self.philox_streams(tcx, instance, mir, args, location, mlir_block, state)?;
        self.uint_to_uniform_float(c0, location, mlir_block).map(Some)
    }

    /// `triton::Triton::randn(seed, offsets, n_rounds)`: standard-normal `f32`, by Box-Muller
    /// on the first two Philox streams (Triton's `pair_uniform_to_normal`).
    pub fn codegen_randn_call<'tcx>(
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
        let [c0, c1, ..] =
            self.philox_streams(tcx, instance, mir, args, location, mlir_block, state)?;
        let u1 = self.uint_to_uniform_float(c0, location, mlir_block)?;
        let u2 = self.uint_to_uniform_float(c1, location, mlir_block)?;
        let ty = u1.r#type();
        let emit = |op: Operation<'a>| self.append_result(op, mlir_block);
        let unary = |name: &str, x: Value<'a, 'a>| {
            OperationBuilder::new(name, location)
                .add_operands(&[x])
                .add_results(&[ty])
                .build()
                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })
        };
        let min_u1 = self.splat_f32(1.0e-7, ty, location, mlir_block)?;
        let u1 = emit(
            OperationBuilder::new("arith.maximumf", location)
                .add_operands(&[min_u1, u1])
                .add_results(&[ty])
                .build()
                .map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?,
        )?;
        let two_pi = self.splat_f32(std::f32::consts::TAU, ty, location, mlir_block)?;
        let theta = emit(melior::dialect::arith::mulf(two_pi, u2, location))?;
        let minus_two = self.splat_f32(-2.0, ty, location, mlir_block)?;
        let log_u1 = emit(unary("math.log", u1)?)?;
        let r = emit(unary("math.sqrt", emit(melior::dialect::arith::mulf(minus_two, log_u1, location))?)?)?;
        let cos_theta = emit(unary("math.cos", theta)?)?;
        emit(melior::dialect::arith::mulf(r, cos_theta, location)).map(Some)
    }

    /// Runs Philox-4x32 (Triton's `randint4x` -> `philox` -> `philox_impl`) and returns the
    /// four `i32` streams. With a 32-bit seed and 32-bit offsets the key is `(seed, 0)` and the
    /// counter `(offsets, 0, 0, 0)`; the `n_rounds` rounds are unrolled, as Triton does.
    fn philox_streams<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        args: &[Spanned<Operand<'tcx>>],
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
        state: &mut CodegenState<'a, 'a>,
    ) -> Result<[Value<'a, 'a>; 4], MlirError> {
        use melior::dialect::arith::{addi, muli, xori};

        const KEY_A: u32 = 0x9E37_79B9;
        const KEY_B: u32 = 0xBB67_AE85;
        const ROUND_A: u32 = 0xD251_1F53;
        const ROUND_B: u32 = 0xCD9E_8D57;

        let fail = |err: String| MlirError::CodegenFailed { err: format!("philox: {err}") };
        if args.len() != 3 {
            return Err(fail(format!("expected 3 arguments, got {}", args.len())));
        }
        let mut operand = |i: usize| {
            self.codegen_operand(
                tcx,
                instance,
                &args[i].node,
                args[i].node.ty(mir, tcx),
                location,
                mlir_block,
                state,
            )
        };
        let seed = operand(0)?;
        let offsets = operand(1)?;
        let n_rounds = self
            .to_scalar_int(tcx, instance, &args[2].node)
            .ok()
            .and_then(|s| u32::try_from(s.to_i32()).ok())
            .ok_or_else(|| fail("`n_rounds` must be a non-negative compile-time constant".into()))?;

        let ty = offsets.r#type();
        let emit = |op: Operation<'a>| self.append_result(op, mlir_block);
        let splat_const = |v: u32| self.splat_i32(v, ty, location, mlir_block);
        let seed_splat: Operation<'a> = splat(self.module.context(), location, seed, ty)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let (mut k0, mut k1) = (emit(seed_splat)?, splat_const(0)?);
        let zero = splat_const(0)?;
        let (mut c0, mut c1, mut c2, mut c3) = (offsets, zero, zero, zero);
        let (round_a, round_b) = (splat_const(ROUND_A)?, splat_const(ROUND_B)?);
        let (key_a, key_b) = (splat_const(KEY_A)?, splat_const(KEY_B)?);
        let umulhi = |x, y| -> Result<Value<'a, 'a>, MlirError> {
            let op: Operation<'a> = mulhiui(self.module.context(), location, x, y)
                .map_err(|e| MlirError::CreateOperation { err: e })?
                .into();
            emit(op)
        };
        for _ in 0..n_rounds {
            let (old_c0, old_c2) = (c0, c2);
            c0 = emit(xori(emit(xori(umulhi(round_b, old_c2)?, c1, location))?, k0, location))?;
            c2 = emit(xori(emit(xori(umulhi(round_a, old_c0)?, c3, location))?, k1, location))?;
            c1 = emit(muli(round_b, old_c2, location))?;
            c3 = emit(muli(round_a, old_c0, location))?;
            k0 = emit(addi(k0, key_a, location))?;
            k1 = emit(addi(k1, key_b, location))?;
        }
        Ok([c0, c1, c2, c3])
    }

    /// Triton's `uint_to_uniform_float` for 32-bit input: fold negative values onto
    /// non-negative ones with `~x`, then scale by the largest factor keeping `MAX_INT * scale`
    /// below 1.0.
    fn uint_to_uniform_float(
        &self,
        x: Value<'a, 'a>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        use melior::dialect::arith::{CmpiPredicate, cmpi, mulf, select, sitofp, xori};

        let context = self.module.context();
        let ty = x.r#type();
        let emit = |op: Operation<'a>| self.append_result(op, mlir_block);
        let zero = self.splat_i32(0, ty, location, mlir_block)?;
        let all_ones = self.splat_i32(u32::MAX, ty, location, mlir_block)?;
        let negative = emit(cmpi(context, CmpiPredicate::Slt, x, zero, location))?;
        let flipped = emit(xori(x, all_ones, location))?;
        let folded = emit(select(negative, flipped, x, location))?;
        let dims = RankedTensorType::try_from(ty).ok().and_then(|t| t.dims().ok()).ok_or_else(|| {
            MlirError::CodegenFailed { err: format!("philox: offsets are not a ranked tensor: {ty}") }
        })?;
        let f32_ty = tensor_type(&dims, melior::ir::r#type::Type::float32(context)).into();
        let as_float = emit(sitofp(folded, f32_ty, location))?;
        let scale = self.splat_f32(4.656_612_734_2e-10, f32_ty, location, mlir_block)?;
        emit(mulf(as_float, scale, location))
    }

    /// Appends `op` and returns its single result.
    fn append_result(
        &self,
        op: Operation<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let op = mlir_block.append_operation(op);
        Ok(op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?.into())
    }

    /// An `i32` constant (given as its bit pattern) splatted to the tensor type `ty`.
    fn splat_i32(
        &self,
        value: u32,
        ty: melior::ir::Type<'a>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let context = self.module.context();
        let scalar: Operation<'a> = create_int_constant(context, location, Int::I32(value))
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let scalar = self.append_result(scalar, mlir_block)?;
        let op: Operation<'a> = splat(context, location, scalar, ty)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        self.append_result(op, mlir_block)
    }

    /// An `f32` constant splatted to the tensor type `ty`.
    fn splat_f32(
        &self,
        value: f32,
        ty: melior::ir::Type<'a>,
        location: Location<'a>,
        mlir_block: &BlockRef<'a, 'a>,
    ) -> Result<Value<'a, 'a>, MlirError> {
        let context = self.module.context();
        let f32_ty = melior::ir::r#type::Type::float32(context);
        let attr = FloatAttribute::new(context, f32_ty, f64::from(value));
        let scalar =
            self.append_result(melior::dialect::arith::constant(context, attr.into(), location), mlir_block)?;
        let op: Operation<'a> = splat(context, location, scalar, ty)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        self.append_result(op, mlir_block)
    }

    /// `triton::Triton::inline_asm_elementwise(asm, constraints, is_pure, pack)`. The
    /// signature has no input tensors and no shape, so the operands and result type of
    /// `tt.elementwise_inline_asm` cannot be derived from it; this is a compile error until
    /// the API takes its inputs.
    pub fn codegen_inline_asm_elementwise_call<'tcx>(
        &self,
        _tcx: TyCtxt<'tcx>,
        _instance: &Instance<'tcx>,
        _mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
        _args: &[Spanned<Operand<'tcx>>],
        _destination: &Place<'tcx>,
        _target: &Option<BasicBlock>,
        _unwind: &UnwindAction,
        _call_source: &CallSource,
        _fn_span: &Span,
        _location: Location<'a>,
        _mlir_block: &BlockRef<'a, 'a>,
        _state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        Err(MlirError::CodegenFailed {
            err: "inline_asm_elementwise is not supported: its signature takes no input \
                  tensors or result shape, so `tt.elementwise_inline_asm` cannot be built"
                .into(),
        })
    }

    /// `triton::Triton::multiple_of` / `max_contiguous` / `max_constancy` —
    /// compiler hint annotations; return the input tensor unchanged.
    pub fn codegen_multiple_of_call<'tcx>(
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
        let v = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        Ok(Some(v))
    }

    pub fn codegen_max_contiguous_call<'tcx>(
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
        let v = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        Ok(Some(v))
    }

    pub fn codegen_max_constancy_call<'tcx>(
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
        let v = self.codegen_operand(
            tcx,
            instance,
            &args[0].node,
            args[0].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;
        Ok(Some(v))
    }

    /// `triton::Triton::num_programs` — `tt.get_num_programs`.
    pub fn codegen_num_programs_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        _mir: &Body<'tcx>,
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
        _state: &mut CodegenState<'a, 'a>,
    ) -> Result<Option<Value<'a, 'a>>, MlirError> {
        let axis_val =
            self.to_scalar_int(tcx, instance, &args[0].node).map(|s| s.to_i32()).unwrap_or(0);
        let axis = ProgramAxis::from(axis_val);
        let op: Operation<'a> = create_get_num_programs(self.module.context(), location, axis)
            .map_err(|e| MlirError::CreateOperation { err: e })?
            .into();
        let result = op.result(0).map_err(|e| MlirError::CodegenFailed { err: e.to_string() })?;
        mlir_block.append_operation(op);
        Ok(Some(result.into()))
    }

    /// `triton::Triton::full` — splat a scalar value to a tensor shape.
    pub fn codegen_full_call<'tcx>(
        &self,
        tcx: TyCtxt<'tcx>,
        instance: &Instance<'tcx>,
        mir: &Body<'tcx>,
        _func: &Operand<'tcx>,
        _func_name: &str,
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
        // args: [shape: &[i32], value: E] — shape comes from args[0] (slice), value from args[1].
        let scalar = self.codegen_operand(
            tcx,
            instance,
            &args[1].node,
            args[1].node.ty(mir, tcx),
            location,
            mlir_block,
            state,
        )?;

        let dest_ty = destination.ty(mir, tcx).ty;
        let dest_ty = instance.instantiate_mir_and_normalize_erasing_regions(
            tcx,
            TypingEnv::fully_monomorphized(),
            EarlyBinder::bind(tcx, dest_ty),
        );
        let mapped_ty = self.type_mapper.map_type(self.module.context(), &tcx, &dest_ty);

        // Try to get the concrete shape from slice_shape[args[0] local] to avoid `tensor<?xE>`.
        let result_ty = if let Operand::Copy(p) | Operand::Move(p) = &args[0].node {
            if let Some(shape) = state.slice_shape.get(&p.local) {
                let elem_ty = if let Ok(t) = RankedTensorType::try_from(mapped_ty) {
                    t.element()
                } else {
                    mapped_ty
                };
                tensor_type(shape, elem_ty).into()
            } else {
                mapped_ty
            }
        } else {
            mapped_ty
        };

        self.splat_scalar_const(location, scalar, result_ty, mlir_block).map(Some)
    }
}
