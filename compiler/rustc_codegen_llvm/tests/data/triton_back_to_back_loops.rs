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

// Two `for` loops with nothing computed between them, both accumulators starting from the same
// `z`. MIR then numbers the block that leaves the first loop and enters the second after the
// second loop's header, and the init-block search only looked below the header, so the second
// loop was never lowered and codegen panicked with "codegen_goto: target bbN not in
// basic_blocks" (teenygrad#156).
fn back_to_back_loops<T: Triton, D: Float, const BLOCK: i32>(n: i32) {
    let ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK]);
    let z = T::zeros::<D>(&[BLOCK]);
    let mut a = z;
    for _ in 0..n {
        a = a + T::load::<D, 1>(ptrs, None, None, &[0], None, None, None, false);
    }
    let mut b = z;
    for _ in 0..n {
        b = b + T::load::<D, 1>(ptrs, None, None, &[0], None, None, None, false) * a;
    }
    T::store::<D, 1>(ptrs, b, None, &[0], None, None);
}

use triton::llvm::triton::num::*;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(n: i32) {
    back_to_back_loops::<LlvmTriton, f32, 16>(n);
}
