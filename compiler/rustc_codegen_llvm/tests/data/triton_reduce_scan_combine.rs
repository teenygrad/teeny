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

// Regression test for teenyc-37e: `T::reduce` and `T::associative_scan` with a user combine
// function. The combine function is passed as a fn pointer, and its body has to be generated
// inline into the single-block tt.reduce / tt.scan combine region with scalar arguments. Different
// combine ops are used for the two calls, and the kernel itself does no other multiply or add, so
// each op's presence in the PTX shows that its combine body was emitted.
fn combine_mul<TT: Triton, DD: Dtype>(lhs: TT::Tensor<DD>, rhs: TT::Tensor<DD>) -> TT::Tensor<DD> {
    lhs * rhs
}

fn combine_add<TT: Triton, DD: Dtype>(lhs: TT::Tensor<DD>, rhs: TT::Tensor<DD>) -> TT::Tensor<DD> {
    lhs + rhs
}

fn reduce_scan_combine<T: Triton, D: Float, const BLOCK_SIZE: i32>(
    x_ptr: T::Pointer<D>,
    output_ptr: T::Pointer<D>,
) {
    let ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    let x = T::load::<D, 1>(ptrs, None, None, &[0], None, None, None, false);
    let prod = T::reduce::<D, D>(x, 0, combine_mul::<T, D>, true);
    let scan = T::associative_scan::<D>(x, 0, combine_add::<T, D>, true);
    T::store::<D, 1>(ptrs, scan / prod, None, &[0], None, None);
    let _ = (x_ptr, output_ptr);
}

use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(x_ptr: *mut f32, output_ptr: *mut f32) {
    let x_ptr = LlvmPointer(x_ptr as *mut _);
    let output_ptr = LlvmPointer(output_ptr as *mut _);
    reduce_scan_combine::<LlvmTriton, f32, 1024>(x_ptr, output_ptr);
}
