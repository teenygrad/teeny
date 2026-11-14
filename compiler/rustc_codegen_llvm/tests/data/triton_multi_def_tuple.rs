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

// teenyc-2m7: a local assigned in more than one place (here, chosen by `first` at run time) must
// not silently take the value of whichever assignment was recorded. A tuple merges field by
// field where the paths meet (teenyc-3a1), so this compiles.
fn multi_def<T: Triton, D: Float>(src: T::Pointer<D>, dst: T::Pointer<D>, n: i32, first: bool) {
    let ptrs = T::zeros::<T::Pointer<D>>(&[16, 2]);
    let x = T::load::<D, 2>(ptrs, None, None, &[0, 1], None, None, None, false);
    // Each arm's call writes the same tuple local; a tuple built from separate values would be
    // split into scalars by MIR optimizations and merged like any other SSA value.
    let pair = if first { T::split(x) } else { T::split(x * x) };
    let out = T::zeros::<T::Pointer<D>>(&[16]);
    T::store::<D, 1>(out, pair.0 - pair.1, None, &[0], None, None);
    let _ = (src, dst, n);
}

use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(src: *mut f32, dst: *mut f32, n: i32, first: bool) {
    multi_def::<LlvmTriton, f32>(LlvmPointer(src as *mut _), LlvmPointer(dst as *mut _), n, first);
}
