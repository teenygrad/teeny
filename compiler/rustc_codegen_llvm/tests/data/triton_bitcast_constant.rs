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

// teenygrad#170: the bits of 1.5f32 are 0x3FC0_0000. Reinterpreted, `>> 16 & 0xFF` gives 0xC0,
// and `<< 1 ^ 3` then 387; converted by value (1), as `cast` did before, the same ops give 3.
// The kernel folds to a constant store, so the assembly shows which one it computed.
fn k<T: Triton, const BLOCK: i32>(out_ptr: T::Pointer<i32>)
where
    T::I32Tensor: Tensor<i32, 1>,
    T::Tensor<i32>: crate::core::ops::Shr<i32, Output = T::Tensor<i32>>
        + crate::core::ops::Shl<i32, Output = T::Tensor<i32>>
        + crate::core::ops::BitAnd<i32, Output = T::Tensor<i32>>
        + crate::core::ops::BitXor<i32, Output = T::Tensor<i32>>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
{
    let bits = T::cast::<f32, i32>(T::full::<f32>(&[BLOCK], 1.5), None, true);
    let value = (((bits >> 16) & 0xFF) << 1) ^ 3;
    T::store(out_ptr.add_offsets(T::arange(0, BLOCK)), value, None, &[], None, None);
}

use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(out_ptr: *mut i32) {
    k::<LlvmTriton, 16>(LlvmPointer(out_ptr as *mut _));
}
