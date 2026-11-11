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

// Regression test for teenyc-1qz: MLIR integers are signless, so signedness has to come from the
// Rust element type, but tensor division, comparison, the max/min reductions and int<->float
// casts always picked the signed op, so u32 tensors miscompiled for values >= 2^31. An int->int
// widening cast was always emitted as a truncation. `n` is a kernel argument, so no operation
// can fold.
fn unsigned_tensor<T: Triton, const BLOCK_SIZE: i32>(n: u32) {
    let u_ptrs = T::zeros::<T::Pointer<u32>>(&[BLOCK_SIZE]);
    let u = T::load::<u32, 1>(u_ptrs, None, None, &[0], None, None, None, false);
    let un = T::full::<u32>(&[BLOCK_SIZE], n);

    // div.u32, setp.lt.u32 (used as the store mask), and max.u32 / min.u32 from the reductions.
    let reduced = T::max(u, Some(0), true) + T::min(u, Some(0), true);
    T::store::<u32, 1>(u_ptrs, u / un + reduced, Some(T::lt(u, un)), &[0], None, None);

    // cvt.rn.f32.u32 and cvt.rzi.u32.f32.
    let f_ptrs = T::zeros::<T::Pointer<f32>>(&[BLOCK_SIZE]);
    let f = T::load::<f32, 1>(f_ptrs, None, None, &[0], None, None, None, false);
    T::store::<f32, 1>(f_ptrs, T::cast::<u32, f32>(u, None, false), None, &[0], None, None);
    T::store::<u32, 1>(u_ptrs, T::cast::<f32, u32>(f, None, false), None, &[0], None, None);

    // Widening casts: cvt.u32.u16 for unsigned, cvt.s32.s16 for signed.
    let h_ptrs = T::zeros::<T::Pointer<u16>>(&[BLOCK_SIZE]);
    let h = T::load::<u16, 1>(h_ptrs, None, None, &[0], None, None, None, false);
    T::store::<u32, 1>(u_ptrs, T::cast::<u16, u32>(h, None, false), None, &[0], None, None);
    let s_ptrs = T::zeros::<T::Pointer<i16>>(&[BLOCK_SIZE]);
    let s = T::load::<i16, 1>(s_ptrs, None, None, &[0], None, None, None, false);
    let i_ptrs = T::zeros::<T::Pointer<i32>>(&[BLOCK_SIZE]);
    T::store::<i32, 1>(i_ptrs, T::cast::<i16, i32>(s, None, false), None, &[0], None, None);
}

type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(n: u32) {
    unsigned_tensor::<LlvmTriton, 1024>(n);
}
