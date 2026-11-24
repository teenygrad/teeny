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

// teenygrad#170: `T::cast(.., true)` ignored its `bitcast` flag and converted by value, so f32
// words carrying packed bytes could not be read back out. Each word is reinterpreted as an i32
// and its third byte extracted with a shift and a mask, as a kernel unpacking quantised weights
// does. Before the fix, the shift lowered to an unshaped `tensor<?xf32>` that failed to verify.
fn k<T: Triton, const BLOCK: i32>(x_ptr: T::Pointer<f32>, y_ptr: T::Pointer<i32>)
where
    T::I32Tensor: Tensor<i32, 1>,
    T::Tensor<i32>: crate::core::ops::Shr<i32, Output = T::Tensor<i32>>
        + crate::core::ops::BitAnd<i32, Output = T::Tensor<i32>>,
    T::Pointer<f32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<f32>>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
{
    let offsets = T::arange(0, BLOCK);
    let x = T::load(x_ptr.add_offsets(offsets), None, None, &[], None, None, None, false);
    let bits = T::cast::<f32, i32>(x, None, true);
    let byte = (bits >> 16) & 0xFF;
    T::store(y_ptr.add_offsets(offsets), byte, None, &[], None, None);
}

use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(x_ptr: *mut f32, y_ptr: *mut i32) {
    k::<LlvmTriton, 16>(LlvmPointer(x_ptr as *mut _), LlvmPointer(y_ptr as *mut _));
}
