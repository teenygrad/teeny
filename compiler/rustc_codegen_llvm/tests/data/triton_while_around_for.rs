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

// A `while` loop stepping by 2 around a `for` loop. The `for` must stay nested inside it, and the
// `while` must keep its step of 2: the inner loop's `Some` payload is not the outer loop's
// variable, so it mustn't make the outer loop look like a unit-step range loop.
fn while_around_for<T: Triton, D: Float, const BLOCK: i32>(n: i32, b: i32) {
    let ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK]);
    let mut acc = T::zeros::<D>(&[BLOCK]);
    let mut i = 0;
    while i < n {
        for j in 0..b {
            acc = acc + T::cast::<i32, D>(T::full::<i32>(&[BLOCK], i + j), None, false);
        }
        i = i + 2;
    }
    T::store::<D, 1>(ptrs, acc, None, &[0], None, None);
}

use triton::llvm::triton::num::*;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(n: i32, b: i32) {
    while_around_for::<LlvmTriton, f32, 16>(n, b);
}
