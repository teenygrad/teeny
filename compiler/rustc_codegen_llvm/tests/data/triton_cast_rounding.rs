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

// teenygrad#170: `T::cast` ignored `fp_downcast_rounding`, so every float narrowing rounded to
// nearest-even. 1/3 lies between two f32s; rounding to nearest-even picks 0x3EAA_AAAB and
// rounding towards zero 0x3EAA_AAAA. The low nibble of each rounding's bits (11 and 10) is
// stored, and the kernel folds to constant stores, so the assembly shows which roundings it
// applied.
fn k<T: Triton, const BLOCK: i32>(nearest_ptr: T::Pointer<i32>, zero_ptr: T::Pointer<i32>)
where
    T::I32Tensor: Tensor<i32, 1>,
    T::Tensor<i32>: crate::core::ops::BitAnd<i32, Output = T::Tensor<i32>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
{
    let third = T::full::<f64>(&[BLOCK], 0.333_333_333_333_333_3);
    let nearest = T::cast::<f64, f32>(third, Some(FpDowncastRounding::Rtne), false);
    let zero = T::cast::<f64, f32>(third, Some(FpDowncastRounding::Rtz), false);
    let offsets = T::arange(0, BLOCK);
    let nearest_nibble = T::cast::<f32, i32>(nearest, None, true) & 0xF;
    let zero_nibble = T::cast::<f32, i32>(zero, None, true) & 0xF;
    T::store(nearest_ptr.add_offsets(offsets), nearest_nibble, None, &[], None, None);
    T::store(zero_ptr.add_offsets(offsets), zero_nibble, None, &[], None, None);
}

use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(nearest_ptr: *mut i32, zero_ptr: *mut i32) {
    k::<LlvmTriton, 16>(LlvmPointer(nearest_ptr as *mut _), LlvmPointer(zero_ptr as *mut _));
}
