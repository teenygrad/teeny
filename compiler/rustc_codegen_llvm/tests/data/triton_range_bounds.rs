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

// Loops whose ranges don't start at zero, with a different range each. The scf.for must run from
// each loop's own start to its own end: the lowering used to start every loop at 0, so `START..n`
// ran n times instead of n - START (teenyc-16j).
fn range_bounds<T: Triton, D: Float, const BLOCK: i32, const START: i32>(n: i32, s: i32, m: i32) {
    let ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK]);
    let mut a = T::zeros::<D>(&[BLOCK]);
    for i in START..n {
        a = a + T::cast::<i32, D>(T::full::<i32>(&[BLOCK], i), None, false);
    }
    for j in s..m {
        a = a + T::cast::<i32, D>(T::full::<i32>(&[BLOCK], j), None, false);
    }
    T::store::<D, 1>(ptrs, a, None, &[0], None, None);
}

use triton::llvm::triton::num::*;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(n: i32, s: i32, m: i32) {
    range_bounds::<LlvmTriton, f32, 16, 3>(n, s, m);
}
