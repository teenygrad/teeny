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

// A loop inside an `if` inside a `while` loop stepping by 2. The `if` must lower to an scf.if in
// the scf.while body, with the `for` loop built in its arm (teenyc-9nm): an `if` in a `while` body
// used to panic because the while loop never recorded its back edge.
fn loop_in_if_in_while<T: Triton, D: Float, const BLOCK: i32>(n: i32, b: i32) {
    let ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK]);
    let mut acc = T::zeros::<D>(&[BLOCK]);
    let mut i = 0;
    while i < n {
        if i > 0 {
            for j in 0..b {
                acc = acc + T::cast::<i32, D>(T::full::<i32>(&[BLOCK], i + j), None, false);
            }
        }
        i = i + 2;
    }
    T::store::<D, 1>(ptrs, acc, None, &[0], None, None);
}

use triton::llvm::triton::num::*;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(n: i32, b: i32) {
    loop_in_if_in_while::<LlvmTriton, f32, 16>(n, b);
}
