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

#![allow(non_camel_case_types)]
#![allow(internal_features)]
#![allow(non_snake_case)]
#![allow(dead_code)]
#![feature(no_core)]
#![feature(intrinsics, lang_items)]
#![feature(arbitrary_self_types)]
#![feature(const_trait_impl)]
#![feature(auto_traits)]
#![feature(rustc_attrs)]
#![no_core]
#![no_implicit_prelude]

// `include!` is defined in `core`, which a `no_core` crate doesn't have, so declare the
// compiler built-in directly.
#[rustc_builtin_macro]
macro_rules! include {
    ($file:expr $(,)?) => {{ /* compiler built-in */ }};
}

include!("triton_prelude.rs");
fn kitchen_sink<T: Triton, D: Float, const BLOCK_SIZE: i32, const USE_BIAS: bool>(
    x_ptr: T::Pointer<D>,
    y_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
    n_elements: i32,
    use_bias: bool,
) {
    fn combine_num<TT: Triton, DD: Dtype>(
        lhs: TT::Tensor<DD>,
        rhs: TT::Tensor<DD>,
    ) -> TT::Tensor<DD> {
        lhs + rhs
    }
    let _pid_x = T::program_id(Axis::X);
    let _pid_y = T::program_id(Axis::Y);
    let _pid_z = T::program_id(Axis::Z);
    let _nprog_x = T::num_programs(Axis::X);
    let r = T::arange(0, BLOCK_SIZE);
    let _ = r + n_elements;
    let _ = r - 1;
    let _ = r * 2;
    let z = T::zeros::<D>(&[BLOCK_SIZE]);
    let zl = T::zeros_like(z);
    let _ = T::full::<D>(&[BLOCK_SIZE], D::from_f64(2.0));
    let casted = T::cast::<D, D>(z, Some(FpDowncastRounding::Rtne), false);
    let _casted_rtz = T::cast::<D, D>(casted, Some(FpDowncastRounding::Rtz), true);
    let cat = T::cat(z, zl, true);
    let (ba, bb) = T::broadcast(cat, cat);
    let bto = T::broadcast_to(ba, &[BLOCK_SIZE]);
    let ex = T::expand_dims(bto, 0);
    let p = T::permute(ex, &[0]);
    let rs = T::reshape(p, &[2 * BLOCK_SIZE], false);
    let tr = T::trans(rs, &[0]);
    let rv = T::ravel(tr, false);
    let vw = T::view(rv, &[BLOCK_SIZE]);
    let jn = T::join(vw, bb);
    let il = T::interleave(jn, jn);
    let (sp0, sp1) = T::split(il);
    let dot = T::dot::<D, D>(sp0, sp1, None, Some(InputPrecision::TF32), Some(1));
    let _dot_tf32x3 = T::dot::<D, D>(dot, dot, None, Some(InputPrecision::TF32x3), None);
    let _dot_ieee = T::dot::<D, D>(dot, dot, None, Some(InputPrecision::IEEE), None);
    let scale = zl;
    let _dot_scaled = T::dot_scaled::<D, D, D>(
        dot,
        scale,
        DotFormat::E4M3,
        dot,
        scale,
        DotFormat::E5M2,
        None,
        true,
    );
    let _ = T::dot_scaled::<D, D, D>(
        dot,
        scale,
        DotFormat::E2M1x2,
        dot,
        scale,
        DotFormat::E2M1x4,
        None,
        false,
    );
    let _ = T::dot_scaled::<D, D, D>(
        dot,
        scale,
        DotFormat::BF16x2,
        dot,
        scale,
        DotFormat::Int8,
        None,
        false,
    );
    let _ = T::dot_scaled::<D, D, D>(
        dot,
        scale,
        DotFormat::UInt8,
        dot,
        scale,
        DotFormat::E4M3,
        None,
        false,
    );
    let block_ptr = T::make_block_ptr(x_ptr, &[BLOCK_SIZE], &[1], &[0], &[BLOCK_SIZE], &[0]);
    let block_ptr2 = T::advance(block_ptr, &[1]);
    let tdesc = T::make_tensor_descriptor(
        y_ptr,
        &[BLOCK_SIZE],
        &[1],
        &[BLOCK_SIZE],
        Some(PaddingOption::Zero),
    );
    let tdesc_nan = T::make_tensor_descriptor(
        y_ptr,
        &[BLOCK_SIZE],
        &[1],
        &[BLOCK_SIZE],
        Some(PaddingOption::Nan),
    );
    let tdv = T::load_tensor_descriptor(tdesc, &[0]);
    T::store_tensor_descriptor(tdesc_nan, &[0], tdv);
    let ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    let loaded = T::load::<D, 1>(
        ptrs,
        None,
        Some(T::zeros::<D>(&[BLOCK_SIZE])),
        &[0],
        Some(PaddingOption::Zero),
        Some(CacheModifier::Ca),
        Some(EvictionPolicy::EvictFirst),
        false,
    );
    let _ = T::load::<D, 1>(
        ptrs,
        None,
        None,
        &[0],
        Some(PaddingOption::Nan),
        Some(CacheModifier::Cg),
        Some(EvictionPolicy::EvictLast),
        true,
    );
    let _ = T::load::<D, 1>(
        ptrs,
        None,
        None,
        &[0],
        None,
        Some(CacheModifier::Cv),
        Some(EvictionPolicy::NoEvict),
        false,
    );
    T::store::<D, 1>(
        ptrs,
        loaded,
        None,
        &[0],
        Some(CacheModifier::Wb),
        Some(EvictionPolicy::NoEvict),
    );
    T::store::<D, 1>(ptrs, loaded, None, &[0], Some(CacheModifier::Cs), None);
    let cond = T::lt(loaded, zl);
    let _ = T::where_(cond, loaded, loaded);
    T::assume(cond);
    T::device_assert(cond, "kitchen_sink", Some(cond));
    let fl = T::flip(loaded, Some(0));
    let _ = T::gather(fl, r, 0);
    let _abs = T::abs(loaded);
    let ceil = T::ceil(loaded);
    let floor = T::floor(ceil);
    let cos = T::cos(floor);
    let sin = T::sin(cos);
    let exp = T::exp(sin);
    let exp2 = T::exp2(exp);
    let log = T::log(exp2);
    let log2 = T::log2(log);
    let rsqrt = T::rsqrt(log2);
    let sig = T::sigmoid(rsqrt);
    let sqrt = T::sqrt(sig);
    let sqrt_rn = T::sqrt_rn(sqrt);
    let erf = T::erf(sqrt_rn);
    let atan = T::atan(erf);
    let smax = T::softmax(atan, Some(0), true, true);
    let mx = T::maximum(smax, smax);
    let mn = T::minimum(mx, smax);
    let cl = T::clamp(mn, smax, mx);
    let fm = T::fma(cl, smax, mx);
    let fd = T::fdiv(fm, smax, true);
    let dr = T::div_rn(fd, smax);
    let _cd = T::cdiv(n_elements, BLOCK_SIZE);
    let _swz = T::swizzle2d(0, 0, BLOCK_SIZE, BLOCK_SIZE, 1);
    let _sum = T::sum(dr, Some(0), true);
    let _max = T::max(dr, None, false);
    let (_maxv, _maxi) = T::max_with_indices(dr, 0, true, false);
    let _min = T::min(dr, Some(0), false);
    let (_minv, _mini) = T::min_with_indices(dr, 0, true, false);
    let _argmax = T::argmax(dr, 0, true, false);
    let _argmin = T::argmin(dr, 0, true, false);
    let _xors = T::xor_sum(T::zeros::<i32>(&[BLOCK_SIZE]), Some(0), false);
    let _cumsum = T::cumsum(dr, 0, false);
    let _cumprod = T::cumprod(dr, 0, true);
    let _sort = T::sort(dr, Some(0), true);
    let _hist = T::histogram(r, BLOCK_SIZE, None);
    let _reduced = T::reduce::<D, D>(dr, 0, combine_num::<T, D>, false);
    let _scan = T::associative_scan::<D>(dr, 0, combine_num::<T, D>, true);
    let aptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    let _ = T::atomic_add(aptrs, dr, None, Some(MemSem::Relaxed), Some(MemScope::Cta));
    let _ = T::atomic_max(aptrs, dr, None, Some(MemSem::Acquire), Some(MemScope::Gpu));
    let _ = T::atomic_min(aptrs, dr, None, Some(MemSem::Release), Some(MemScope::Sys));
    let _ = T::atomic_xchg(aptrs, dr, None, Some(MemSem::AcqRel), Some(MemScope::Gpu));
    let _ = T::atomic_cas(aptrs, dr, dr, Some(MemSem::AcqRel), Some(MemScope::Gpu));
    let iptrs = T::zeros::<T::Pointer<i32>>(&[BLOCK_SIZE]);
    let ival = T::zeros::<i32>(&[BLOCK_SIZE]);
    let _ = T::atomic_and(iptrs, ival, None, Some(MemSem::Relaxed), Some(MemScope::Cta));
    let _ = T::atomic_or(iptrs, ival, None, Some(MemSem::Acquire), Some(MemScope::Gpu));
    let _ = T::atomic_xor(iptrs, ival, None, Some(MemSem::Release), Some(MemScope::Sys));
    let u32a = T::zeros::<u32>(&[BLOCK_SIZE]);
    let u32b = T::zeros::<u32>(&[BLOCK_SIZE]);
    let _umulhi = T::umulhi(u32a, u32b);
    let _rand = T::rand(123, r, 10);
    let _randn = T::randn(123, r, 10);
    let _randi = T::randint(123, r, 10);
    let _rand4 = T::randint4x(123, r, 10);
    let mo = T::multiple_of(dr, &[1]);
    let mc = T::max_contiguous(mo, &[1]);
    let mconst = T::max_constancy(mc, &[1]);
    T::debug_barrier();
    T::device_print("val=", mconst, false);
    T::static_assert(true, "BLOCK_SIZE must be positive");
    T::static_print("kitchen_sink");
    // Comparison ops (Triton::gt/ge/lt/le/eq/ne) — exercised on float tensors so the
    // codegen emits arith.cmpf rather than falling through to tt.call.
    let gt_mask = T::gt(mconst, zl);
    let ge_mask = T::ge(mconst, zl);
    let lt_mask = T::lt(mconst, zl);
    let le_mask = T::le(mconst, zl);
    let eq_mask = T::eq(mconst, zl);
    let ne_mask = T::ne(mconst, zl);
    // where_ with a real comparison result (full relu-backward pattern):
    // arith.select requires the condition type to be tensor<Nxi1>, not tensor<?xi1>.
    let _relu_bwd = T::where_(gt_mask, mconst, zl);
    let _ge_sel = T::where_(ge_mask, mconst, zl);
    let _lt_sel = T::where_(lt_mask, mconst, zl);
    let _le_sel = T::where_(le_mask, mconst, zl);
    let _eq_sel = T::where_(eq_mask, mconst, zl);
    let _ne_sel = T::where_(ne_mask, mconst, zl);
    // Scalar comparison ops — tensor vs scalar constant (second arg is a scalar D value).
    let scalar_val = D::from_f64(0.5);
    let gt_s = T::gt_scalar(mconst, scalar_val);
    let ge_s = T::ge_scalar(mconst, scalar_val);
    let lt_s = T::lt_scalar(mconst, scalar_val);
    let le_s = T::le_scalar(mconst, scalar_val);
    let eq_s = T::eq_scalar(mconst, scalar_val);
    let ne_s = T::ne_scalar(mconst, scalar_val);
    let _gt_s_sel = T::where_(gt_s, mconst, zl);
    let _ge_s_sel = T::where_(ge_s, mconst, zl);
    let _lt_s_sel = T::where_(lt_s, mconst, zl);
    let _le_s_sel = T::where_(le_s, mconst, zl);
    let _eq_s_sel = T::where_(eq_s, mconst, zl);
    let _ne_s_sel = T::where_(ne_s, mconst, zl);
    // SwitchInt on a const bool generic (compile-time constant discriminant).
    if USE_BIAS {
        let _ = loaded + z;
    }
    // SwitchInt on a runtime bool SSA value (bool function parameter).
    if use_bias {
        let _ = loaded + z;
    }

    // Range-based for loop: scf.for with runtime bound (n_elements).
    // Accumulates z into for_acc over n_elements iterations.
    let mut for_acc = T::zeros::<D>(&[BLOCK_SIZE]);
    for _k in 0..n_elements {
        for_acc = for_acc + z;
    }
    let _ = for_acc;

    // While loop: scf.while with loop-carried scalar (n_start: i32) and tensor (while_acc).
    // Models the tile-loop pattern used in batchnorm / layernorm kernels.
    let mut while_acc = T::zeros::<D>(&[BLOCK_SIZE]);
    let mut n_start: i32 = 0;
    while n_start < n_elements {
        while_acc = while_acc + z;
        n_start += BLOCK_SIZE;
    }
    let _ = while_acc;

    let out_ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    let _ = T::make_block_ptr(output_ptr, &[BLOCK_SIZE], &[1], &[0], &[BLOCK_SIZE], &[0]);
    T::store::<D, 1>(out_ptrs, mconst, None, &[0], None, None);
    let _ = block_ptr2;

    let N = BLOCK_SIZE;
    let x_ptrs = T::zeros::<T::Pointer<D>>(&[N]);
    let x = T::load::<D, 1>(x_ptrs, None, None, &[0], None, None, None, false);
    let s = T::sqrt(T::sum(x * x, Some(0), true));
    let eps = T::full::<D>(&[N], D::from_f64(1e-6));
    T::store::<D, 1>(out_ptrs, x / T::maximum(s, eps), None, &[0], None, None);
    T::store::<D, 1>(out_ptrs, x / T::minimum(s, eps), None, &[0], None, None);
}

use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(
    x_ptr: *mut f32,
    y_ptr: *mut f32,
    output_ptr: *mut f32,
    n_elements: i32,
) {
    let x_ptr = LlvmPointer(x_ptr as *mut _);
    let y_ptr = LlvmPointer(y_ptr as *mut _);
    let output_ptr = LlvmPointer(output_ptr as *mut _);
    kitchen_sink::<LlvmTriton, f32, 1024, true>(x_ptr, y_ptr, output_ptr, n_elements, false);
}
