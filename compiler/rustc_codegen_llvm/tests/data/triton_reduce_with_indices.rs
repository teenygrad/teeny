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

// Regression test for teenyc-1at: argmax / argmin / max_with_indices / min_with_indices were
// stubs that emitted ub.poison. Each reduces the same loaded tensor; max uses tie_break_left
// (adding the index compare to the combine) and min does not. Every result is stored so none of
// the reductions can be removed as dead.
fn reduce_with_indices<T: Triton, D: Float, const BLOCK_SIZE: i32>()
where
    T::I32Tensor: Into<T::Tensor<i32>>,
{
    let ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    let x = T::load::<D, 1>(ptrs, None, None, &[0], None, None, None, false);

    let (max_v, max_i) = T::max_with_indices(x, 0, true, true);
    let (min_v, min_i) = T::min_with_indices(x, 0, false, true);
    let argmax: T::Tensor<i32> = T::argmax(x, 0, true, true).into();
    let argmin: T::Tensor<i32> = T::argmin(x, 0, false, true).into();
    let max_i: T::Tensor<i32> = max_i.into();
    let min_i: T::Tensor<i32> = min_i.into();

    // The keep_dims results are tensor<1x..>; combining them with full-size tensors broadcasts
    // them, and the full-size stores are what keep them alive.
    T::store::<D, 1>(ptrs, x / (max_v + min_v), None, &[0], None, None);
    let i_ptrs = T::zeros::<T::Pointer<i32>>(&[BLOCK_SIZE]);
    let i = T::load::<i32, 1>(i_ptrs, None, None, &[0], None, None, None, false);
    T::store::<i32, 1>(i_ptrs, i + (max_i + min_i + argmax + argmin), None, &[0], None, None);

    // 2-D, reducing the inner axis: the index operand has to be expanded and broadcast to
    // [BLOCK_SIZE, 4], and the result keeps the outer axis.
    let ptrs2 = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE, 4]);
    let x2 = T::load::<D, 2>(ptrs2, None, None, &[0, 1], None, None, None, false);
    let argmax2: T::Tensor<i32> = T::argmax(x2, 1, true, false).into();
    T::store::<i32, 1>(i_ptrs, i + argmax2, None, &[0], None, None);
}

type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point() {
    reduce_with_indices::<LlvmTriton, f32, 1024>();
}
