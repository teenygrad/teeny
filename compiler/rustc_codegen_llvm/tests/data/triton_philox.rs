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

// Regression test for teenyc-1k4: T::rand / randn / randint / randint4x were stubs that emitted
// ub.poison. The seed is a kernel argument so the Philox rounds cannot constant-fold, and every
// result is stored.
fn philox<T: Triton, const BLOCK_SIZE: i32>(seed: u32)
where
    T::I32Tensor: Into<T::Tensor<i32>>,
{
    let offsets = T::arange(0, BLOCK_SIZE);
    let uniform = T::rand(seed, offsets, 10);
    let normal = T::randn(seed, offsets, 10);
    let int: T::Tensor<i32> = T::randint(seed, offsets, 10).into();
    let (r0, r1, r2, r3) = T::randint4x(seed, offsets, 7);
    let (r0, r1): (T::Tensor<i32>, T::Tensor<i32>) = (r0.into(), r1.into());
    let (r2, r3): (T::Tensor<i32>, T::Tensor<i32>) = (r2.into(), r3.into());

    let f_ptrs = T::zeros::<T::Pointer<f32>>(&[BLOCK_SIZE]);
    T::store::<f32, 1>(f_ptrs, uniform + normal, None, &[0], None, None);
    let i_ptrs = T::zeros::<T::Pointer<i32>>(&[BLOCK_SIZE]);
    T::store::<i32, 1>(i_ptrs, int + r0 + r1 + r2 + r3, None, &[0], None, None);
}

type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(seed: u32) {
    philox::<LlvmTriton, 1024>(seed);
}
