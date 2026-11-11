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

// Regression test for teenyc-3vx: T::maximum emitted arith.maximumf for every element type, and
// T::minimum chose the signed integer op for unsigned tensors too. MLIR integers are signless, so
// the op has to come from the Rust element type: signed and unsigned tensors are both covered, and
// each is compared against a constant so the min/max cannot fold away.
fn int_minmax<T: Triton, const BLOCK_SIZE: i32>() {
    let i_ptrs = T::zeros::<T::Pointer<i32>>(&[BLOCK_SIZE]);
    let i = T::load::<i32, 1>(i_ptrs, None, None, &[0], None, None, None, false);
    let ic = T::full::<i32>(&[BLOCK_SIZE], 3);
    T::store::<i32, 1>(i_ptrs, T::maximum(i, ic) + T::minimum(i, ic), None, &[0], None, None);

    let u_ptrs = T::zeros::<T::Pointer<u32>>(&[BLOCK_SIZE]);
    let u = T::load::<u32, 1>(u_ptrs, None, None, &[0], None, None, None, false);
    let uc = T::full::<u32>(&[BLOCK_SIZE], 3);
    T::store::<u32, 1>(u_ptrs, T::maximum(u, uc) + T::minimum(u, uc), None, &[0], None, None);
}

type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point() {
    int_minmax::<LlvmTriton, 1024>();
}
