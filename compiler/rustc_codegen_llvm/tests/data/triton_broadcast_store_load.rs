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

// Regression test for teenyc-yg9 (teenygrad#119 bug 2): T::store and T::load didn't broadcast a
// store's value or mask, or a load's mask or `other`, to the pointer tensor's shape, as Triton's
// frontend does. `s` is a keep_dims reduction (`tensor<1x..>`, or `tensor<BLOCK_SIZEx1x..>` in 2-D)
// and `lane0` a one-element mask, each used against full-width pointer tensors.
fn broadcast_store_load<T: Triton<I32Tensor = <T as Triton>::Tensor<i32>>, D: Float, const BLOCK_SIZE: i32>() {
    let ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    let x = T::load::<D, 1>(ptrs, None, None, &[0], None, None, None, false);
    let i = T::arange(0, BLOCK_SIZE);
    let s = T::sum(x, Some(0), true);

    // The issue's spelling: "reduce with keep_dims, store lane 0".
    T::store::<D, 1>(ptrs, s, Some(T::lt_scalar(i, 1)), &[0], None, None);
    // The same without a mask.
    T::store::<D, 1>(ptrs, s, None, &[0], None, None);
    // A one-element mask on a full-width value.
    let lane0 = T::lt_scalar(T::arange(0, 1), 1);
    T::store::<D, 1>(ptrs, x, Some(lane0), &[0], None, None);
    // A one-element mask and `other` on a load.
    let y = T::load::<D, 1>(ptrs, Some(lane0), Some(s), &[0], None, None, None, false);
    T::store::<D, 1>(ptrs, y, None, &[0], None, None);

    // 2-D: a row reduction (`tensor<BLOCK_SIZEx1x..>`) stored across every column.
    let ptrs2 = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE, 4]);
    let x2 = T::load::<D, 2>(ptrs2, None, None, &[0, 1], None, None, None, false);
    T::store::<D, 2>(ptrs2, T::sum(x2, Some(1), true), None, &[0, 1], None, None);
}

type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point() {
    broadcast_store_load::<LlvmTriton, f32, 256>();
}
