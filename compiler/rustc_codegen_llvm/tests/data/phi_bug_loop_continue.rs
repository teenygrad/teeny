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

// Minimal reproducer for: codegen_goto: phi local not in ssa_values at branch to bbN
//
// A `continue` statement inside a `for` loop in a Triton kernel triggers a
// phi-node tracking bug in the custom Triton codegen backend.  When the loop
// body has an early-exit via `continue`, the backend fails to include the
// loop-carried accumulator in the phi nodes at the loop header.
fn phi_bug_loop_continue<T: Triton, D: Float, const BLOCK: i32>(n: i32, limit: i32) {
    let z = T::zeros::<D>(&[BLOCK]);
    let mut acc = z;
    for i in 0..n {
        if i >= limit {
            continue;
        }
        acc = acc + z;
    }
    let out_ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK]);
    T::store::<D, 1>(out_ptrs, acc, None, &[0], None, None);
}

use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(n: i32, limit: i32) {
    phi_bug_loop_continue::<LlvmTriton, f32, 64>(n, limit);
}
