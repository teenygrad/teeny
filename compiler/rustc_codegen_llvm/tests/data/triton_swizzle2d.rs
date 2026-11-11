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

// Regression test for teenyc-3er: T::swizzle2d was a stub that emitted ub.poison. Its inputs are
// kernel arguments so the integer arithmetic cannot constant-fold, and both outputs are stored.
fn swizzle2d<T: Triton, const BLOCK_SIZE: i32>(i: i32, j: i32, size_i: i32, size_j: i32, size_g: i32) {
    let (new_i, new_j) = T::swizzle2d(i, j, size_i, size_j, size_g);
    let ptrs = T::zeros::<T::Pointer<i32>>(&[BLOCK_SIZE]);
    let x = T::load::<i32, 1>(ptrs, None, None, &[0], None, None, None, false);
    let swizzled = T::full::<i32>(&[BLOCK_SIZE], new_i) + T::full::<i32>(&[BLOCK_SIZE], new_j);
    T::store::<i32, 1>(ptrs, x + swizzled, None, &[0], None, None);
}

type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(i: i32, j: i32, size_i: i32, size_j: i32, size_g: i32) {
    swizzle2d::<LlvmTriton, 1024>(i, j, size_i, size_j, size_g);
}
