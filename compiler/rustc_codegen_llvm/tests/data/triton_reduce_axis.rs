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

// Regression test for teenyc-26l / teenyc-z81: the Option<i32> axis of sum/max/min/xor_sum and
// softmax's dim were misread (None and non-constant Some became axis 0, and softmax ignored dim),
// and keep_dims was dropped for N-D sources. Each case's result is combined with a full-size
// tensor and stored, so its shape has to be exactly what Triton produces.
fn reduce_axis<T: Triton, D: Float, const BLOCK_SIZE: i32>() {
    let cols = 4;
    let ptrs1 = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    let ptrs2 = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE, cols]);
    let x1 = T::load::<D, 1>(ptrs1, None, None, &[0], None, None, None, false);
    let x2 = T::load::<D, 2>(ptrs2, None, None, &[0, 1], None, None, None, false);

    // Some(1) on 2-D without keep_dims: [BLOCK_SIZE].
    T::store::<D, 1>(ptrs1, x1 + T::sum(x2, Some(1), false), None, &[0], None, None);
    // Negative axis with keep_dims on 1-D: [1].
    T::store::<D, 1>(ptrs1, x1 * T::min(x1, Some(-1), true), None, &[0], None, None);
    // None with keep_dims on 2-D: [1, 1].
    T::store::<D, 2>(ptrs2, x2 + T::max(x2, None, true), None, &[0, 1], None, None);
    // Some(0) with keep_dims on 2-D: [1, cols].
    T::store::<D, 2>(ptrs2, x2 + T::sum(x2, Some(0), true), None, &[0, 1], None, None);

    let i_ptrs = T::zeros::<T::Pointer<i32>>(&[BLOCK_SIZE, cols]);
    let i2 = T::load::<i32, 2>(i_ptrs, None, None, &[0, 1], None, None, None, false);
    // None with keep_dims on a 2-D integer tensor: [1, 1].
    T::store::<i32, 2>(i_ptrs, i2 + T::xor_sum(i2, None, true), None, &[0, 1], None, None);

    // softmax along the inner axis of a 2-D tensor, and along the default axis (0) of a 1-D one.
    T::store::<D, 2>(ptrs2, T::softmax(x2, Some(1), true, false), None, &[0, 1], None, None);
    T::store::<D, 1>(ptrs1, T::softmax(x1, None, false, false), None, &[0], None, None);
}

type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point() {
    reduce_axis::<LlvmTriton, f32, 256>();
}
