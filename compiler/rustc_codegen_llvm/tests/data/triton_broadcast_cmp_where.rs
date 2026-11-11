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

// Regression test for teenyc-39u (teenygrad#119 bug 1): comparisons and T::where_ didn't
// broadcast operands of different shapes. `s` keeps its reduced dimension, so it is `tensor<1x..>`
// while `x` and `eps` are `tensor<BLOCK_SIZEx..>`. The narrow operand is put on each side of a
// comparison, and in each position of where_: condition, true value and false value.
fn broadcast_cmp_where<T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>, D: Float, const BLOCK_SIZE: i32>() {
    let ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    let x = T::load::<D, 1>(ptrs, None, None, &[0], None, None, None, false);
    let s = T::sqrt(T::sum(x * x, Some(0), true));
    let eps = T::zeros_like(x);

    // F.normalize's clamp_min: narrow operand on the left of the comparison and as the false value.
    T::store::<D, 1>(ptrs, x / T::where_(T::lt(s, eps), eps, s), None, &[0], None, None);
    // Narrow operand on the right, and as the true value.
    T::store::<D, 1>(ptrs, T::where_(T::gt(eps, s), s, eps), None, &[0], None, None);
    T::store::<D, 1>(ptrs, T::where_(T::le(x, s), x, eps), None, &[0], None, None);
    T::store::<D, 1>(ptrs, T::where_(T::ge(s, x), x, eps), None, &[0], None, None);
    T::store::<D, 1>(ptrs, T::where_(T::eq(s, x), x, eps), None, &[0], None, None);
    T::store::<D, 1>(ptrs, T::where_(T::ne(x, s), x, eps), None, &[0], None, None);
    // Narrow condition with full-width values.
    T::store::<D, 1>(ptrs, T::where_(T::lt(s, s), x, eps), None, &[0], None, None);

    // Integer comparison (arith.cmpi) against a reduction.
    let i = T::arange(0, BLOCK_SIZE);
    let m = T::max(i, Some(0), true);
    let i_ptrs = T::zeros::<T::Pointer<i32>>(&[BLOCK_SIZE]);
    T::store::<i32, 1>(i_ptrs, T::where_(T::lt(i, m), i, m), None, &[0], None, None);
}

type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point() {
    broadcast_cmp_where::<LlvmTriton, f32, 1024>();
}
