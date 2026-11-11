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

// Regression test for teenyc-3bk: `T::load` given `other` but no mask. tt.load only accepts
// `other` together with a mask (its verifier checks the mask slot whenever there is more than one
// operand), so the load used to fail verification with "mask type matches ptr type".
fn load_other_without_mask<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
) {
    let ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    let x = T::load::<D, 1>(
        ptrs,
        None,
        Some(T::zeros::<D>(&[BLOCK_SIZE])),
        &[0],
        None,
        None,
        None,
        false,
    );
    T::store::<D, 1>(ptrs, x + x, None, &[0], None, None);
    let _ = (x_ptr, output_ptr);
}

use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(x_ptr: *mut f32, output_ptr: *mut f32) {
    let x_ptr = LlvmPointer(x_ptr as *mut _);
    let output_ptr = LlvmPointer(output_ptr as *mut _);
    load_other_without_mask::<LlvmTriton, f32, 1024>(x_ptr, output_ptr);
}
