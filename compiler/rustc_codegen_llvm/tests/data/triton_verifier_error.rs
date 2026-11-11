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

// Regression test for teenyc-1nq: a verifier failure has to be reported with the op, the
// verifier's message and the source location, not just "failed to run pass". `T::join` doesn't
// check that its operands have the same shape, so joining a 1024- and a 64-element tensor builds
// a `tt.join` that MLIR's verifier rejects. If join starts rejecting the mismatch itself, this
// needs another way to reach the verifier.
fn verifier_error<T: Triton, D: Float, const BLOCK_SIZE: i32>() {
    let ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE]);
    let x = T::load::<D, 1>(ptrs, None, None, &[0], None, None, None, false);
    let small = T::zeros::<T::Pointer<D>>(&[64]);
    let y = T::load::<D, 1>(small, None, None, &[0], None, None, None, false);
    let joined = T::join(x, y);
    let out = T::zeros::<T::Pointer<D>>(&[BLOCK_SIZE, 2]);
    T::store::<D, 2>(out, joined, None, &[0, 1], None, None);
}

type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point() {
    verifier_error::<LlvmTriton, f32, 1024>();
}
