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

// A `while` loop that counts down by a constant step, as teeny-kernels' maxpool2d clamps its row
// bound with `while kh_hi > KH { kh_hi -= 1 }`. scf::upliftWhileToForLoop only matches loops that
// count up, so this stayed an scf.while, which Triton doesn't pipeline (teenyc-1zz). As in
// maxpool2d, the count starts from what a counting-up loop before it leaves.
fn k<T: Triton, const BLOCK: i32>(out_ptr: T::Pointer<i32>, n: i32, m: i32)
where
    T::I32Tensor: Tensor<i32, 1>,
    T::Pointer<i32>: AddOffsets<i32, 1, T::I32Tensor, Output = T::Tensor<T::Pointer<i32>>>,
{
    let mut lo: i32 = 0;
    let mut start: i32 = n - 3;
    while start < 0 {
        lo += 1;
        start += 1;
    }
    let mut hi: i32 = lo + (m - start);
    let mut acc = T::full::<i32>(&[BLOCK], -100);
    while hi > 5 {
        acc = T::maximum(acc, T::full::<i32>(&[BLOCK], hi));
        hi -= 3;
    }
    let ptrs = out_ptr.add_offsets(T::arange(0, BLOCK));
    T::store(ptrs, acc, None, &[], None, None);
}

use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(out_ptr: *mut i32, n: i32, m: i32) {
    k::<LlvmTriton, 16>(LlvmPointer(out_ptr as *mut _), n, m);
}
