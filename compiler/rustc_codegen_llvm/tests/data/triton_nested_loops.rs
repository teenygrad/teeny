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

// A `for` loop nested in another, both carrying an accumulator. The inner loop used to become an
// `scf.if (B > 0)` around a single copy of its body with `j` fixed at 0 (teenygrad#155), so it
// must lower to an scf.for of its own inside the outer one.
fn nested_loops<T: Triton, D: Float, const BLOCK: i32>(a: i32, b: i32) {
    let ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK]);
    let mut acc = T::zeros::<D>(&[BLOCK]);
    for i in 0..a {
        let mut row = T::zeros::<D>(&[BLOCK]);
        for j in 0..b {
            row = row + T::cast::<i32, D>(T::full::<i32>(&[BLOCK], i * b + j), None, false);
        }
        acc = acc + row;
    }
    T::store::<D, 1>(ptrs, acc, None, &[0], None, None);
}

use triton::llvm::triton::num::*;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(a: i32, b: i32) {
    nested_loops::<LlvmTriton, f32, 16>(a, b);
}
