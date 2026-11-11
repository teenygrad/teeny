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

// Regression test for teenyc-2o5: a scalar `as` cast between integer types returned its operand
// unchanged, so the value kept the source width: widening never extended and narrowing never
// truncated. Each result is splatted into a tensor of its Rust type, which only verifies when the
// scalar has that width. `s` and `u` are kernel arguments, so no cast can fold.
fn scalar_int_casts<T: Triton, const BLOCK_SIZE: i32>(s: i32, u: u32) {
    // Sign extension (cvt.s64.s32) and zero extension (cvt.u64.u32). The signed operand is
    // computed: a bare argument's extension folds into a sign-extending parameter load.
    let i64_ptrs = T::zeros::<T::Pointer<i64>>(&[BLOCK_SIZE]);
    T::store::<i64, 1>(i64_ptrs, T::full::<i64>(&[BLOCK_SIZE], (s + 1) as i64), None, &[0], None, None);
    let u64_ptrs = T::zeros::<T::Pointer<u64>>(&[BLOCK_SIZE]);
    T::store::<u64, 1>(u64_ptrs, T::full::<u64>(&[BLOCK_SIZE], u as u64), None, &[0], None, None);
    // Truncation.
    let u8_ptrs = T::zeros::<T::Pointer<u8>>(&[BLOCK_SIZE]);
    T::store::<u8, 1>(u8_ptrs, T::full::<u8>(&[BLOCK_SIZE], u as u8), None, &[0], None, None);
    // Same width, different signedness: a no-op on signless integers.
    let i32_ptrs = T::zeros::<T::Pointer<i32>>(&[BLOCK_SIZE]);
    T::store::<i32, 1>(i32_ptrs, T::full::<i32>(&[BLOCK_SIZE], u as i32), None, &[0], None, None);
}

type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(s: i32, u: u32) {
    scalar_int_casts::<LlvmTriton, 1024>(s, u);
}
