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

// Regression test for teenyc-38c. Both `T::zeros` calls take the same shape array, so at
// opt-level 3 MIR GVN rewrites the second `[n]` into `copy` of the first. The shape goes through a
// local so that it is built as an array aggregate rather than promoted to a constant. A
// constant array lives only in `const_arrays` (it has no SSA value), so that copy must carry the
// side-table entry across instead of looking the source up in `ssa_values`.
fn const_array_copy<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
) {
    let n = BLOCK_SIZE;
    let x_ptrs = T::zeros::<T::Pointer<D>>(&[n]);
    let out_ptrs = T::zeros::<T::Pointer<D>>(&[n]);
    let x = T::load::<D, 1>(x_ptrs, None, None, &[0], None, None, None, false);
    T::store::<D, 1>(out_ptrs, x + x, None, &[0], None, None);
    let _ = (x_ptr, output_ptr);
}

use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(x_ptr: *mut f32, output_ptr: *mut f32) {
    let x_ptr = LlvmPointer(x_ptr as *mut _);
    let output_ptr = LlvmPointer(output_ptr as *mut _);
    const_array_copy::<LlvmTriton, f32, 1024>(x_ptr, output_ptr);
}
