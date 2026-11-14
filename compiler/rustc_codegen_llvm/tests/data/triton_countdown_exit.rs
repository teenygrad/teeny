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

// A counter read after a `while` loop that counts down by 3. It runs 20, 17, 14, 11, 8 and stops
// at 5, the first value that fails `i > 6`; an `scf.for` rebuilt from the loop must leave it there
// (teenyc-1zz).
fn k<T: Triton, const BLOCK: i32>(out_ptr: T::Pointer<i32>)
where
    T::I32Tensor: Tensor<i32, 1>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
{
    let mut i: i32 = 20;
    while i > 6 {
        i -= 3;
    }
    let ptrs = out_ptr.add_offsets(T::arange(0, BLOCK));
    T::store(ptrs, T::full::<i32>(&[BLOCK], i), None, &[], None, None);
}

use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(out_ptr: *mut i32) {
    k::<LlvmTriton, 16>(LlvmPointer(out_ptr as *mut _));
}
