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

// An Option chosen by an `if`: assigned `Some` on one path and `None` on the other, so its
// variant is only known at run time where the paths meet (teenyc-3a1).
fn k<T: Triton, D: Float, const BLOCK: i32>(c: i32, n: i32) {
    let ptrs = T::zeros::<T::Pointer<i32>>(&[BLOCK]);
    let m = if c > 0 { Some(n) } else { None };
    let v = match m {
        Some(v) => v * 3,
        None => 7,
    };
    T::store::<i32, 1>(ptrs, T::full::<i32>(&[BLOCK], v), None, &[0], None, None);
}

use triton::llvm::triton::num::*;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(c: i32, n: i32) {
    k::<LlvmTriton, f32, 16>(c, n);
}
