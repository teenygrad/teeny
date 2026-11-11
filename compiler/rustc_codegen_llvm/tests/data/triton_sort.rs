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

// Regression test for teenyc-190: T::sort was a stub that emitted ub.poison. Sorts a 1-D f32
// tensor ascending along an explicit axis and a 2-D i32 tensor descending along its default (last)
// axis, and stores both.
fn sort<T: Triton, D: Float, const BLOCK_SIZE: i32>() {
    let ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    let x = T::load::<D, 1>(ptrs, None, None, &[0], None, None, None, false);
    T::store::<D, 1>(ptrs, T::sort(x, Some(0), false), None, &[0], None, None);

    let rows = 4;
    let ptrs2 = T::zeros::<T::Pointer<i32>>(&[rows, BLOCK_SIZE]);
    let x2 = T::load::<i32, 2>(ptrs2, None, None, &[0, 1], None, None, None, false);
    T::store::<i32, 2>(ptrs2, T::sort(x2, None, true), None, &[0, 1], None, None);
}

type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point() {
    sort::<LlvmTriton, f32, 256>();
}
