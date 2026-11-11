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

// Regression test for teenyc-1jf: Triton 3.6 dropped `tt.make_tensor_ptr` / `tt.advance` from the
// IR (block pointers became frontend-only), so `T::make_block_ptr` and `T::advance` must not emit
// them. The block pointer is two-dimensional with runtime shape, stride and offset entries, and is
// advanced twice, so every argument form the side table records is exercised.
fn block_ptr<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
    n: i32,
) {
    let pid = T::program_id(Axis::X);
    let block_start = pid * BLOCK_SIZE;
    let bp = T::make_block_ptr(x_ptr, &[n, n], &[n, 1], &[block_start, 0], &[BLOCK_SIZE, 1], &[1, 0]);
    let bp = T::advance(bp, &[BLOCK_SIZE, 0]);
    let _bp = T::advance(bp, &[0, block_start]);

    let ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    let x = T::load::<D, 1>(ptrs, None, None, &[0], None, None, None, false);
    T::store::<D, 1>(ptrs, x + x, None, &[0], None, None);
    let _ = output_ptr;
}

use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(x_ptr: *mut f32, output_ptr: *mut f32, n: i32) {
    let x_ptr = LlvmPointer(x_ptr as *mut _);
    let output_ptr = LlvmPointer(output_ptr as *mut _);
    block_ptr::<LlvmTriton, f32, 1024>(x_ptr, output_ptr, n);
}
