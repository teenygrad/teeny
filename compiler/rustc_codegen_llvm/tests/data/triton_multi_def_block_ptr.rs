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

// teenyc-2m7: the backend keeps values with no SSA form in side tables keyed by MIR local, one
// entry per local. A local assigned in more than one place (here, chosen by `first` at run time)
// can't be represented that way, so it must be a compile error rather than silently using the
// value of whichever assignment was recorded.
// A block pointer (`block_ptrs`).
fn multi_def<T: Triton, D: Float>(src: T::Pointer<D>, dst: T::Pointer<D>, n: i32, first: bool) {
    let bp = if first {
        T::make_block_ptr(src, &[n, n], &[n, 1], &[0, 0], &[16, 16], &[1, 0])
    } else {
        T::make_block_ptr(src, &[n, n], &[n, 1], &[16, 0], &[16, 16], &[1, 0])
    };
    let _bp = T::advance(bp, &[16, 0]);
    let _ = dst;
}

use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(src: *mut f32, dst: *mut f32, n: i32, first: bool) {
    multi_def::<LlvmTriton, f32>(LlvmPointer(src as *mut _), LlvmPointer(dst as *mut _), n, first);
}
