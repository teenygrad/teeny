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

// Regression test for teenyc-1qz: scalar u32 division, remainder, comparison and int<->float
// casts were emitted with the signed op. `n`, `m` and `x` are kernel arguments, so no operation
// can fold.
fn unsigned_scalar<T: Triton, const BLOCK_SIZE: i32>(n: u32, m: u32, x: f32) {
    // div.u32, rem.u32 and setp.lt.u32. Each arm's op must survive knowing `n < m` (or not):
    // `n / m` there is known to be 0, which LLVM folds away.
    let q = if n < m { m / n } else { n % m };
    let u_ptrs = T::zeros::<T::Pointer<u32>>(&[BLOCK_SIZE]);
    T::store::<u32, 1>(u_ptrs, T::full::<u32>(&[BLOCK_SIZE], q), None, &[0], None, None);
    // cvt.rzi.u32.f32 and cvt.rn.f32.u32.
    T::store::<u32, 1>(u_ptrs, T::full::<u32>(&[BLOCK_SIZE], x as u32), None, &[0], None, None);
    let f_ptrs = T::zeros::<T::Pointer<f32>>(&[BLOCK_SIZE]);
    T::store::<f32, 1>(f_ptrs, T::full::<f32>(&[BLOCK_SIZE], n as f32), None, &[0], None, None);
}

type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(n: u32, m: u32, x: f32) {
    unsigned_scalar::<LlvmTriton, 1024>(n, m, x);
}
