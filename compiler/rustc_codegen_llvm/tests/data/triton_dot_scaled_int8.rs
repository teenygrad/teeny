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

// `DotFormat::Int8` has no `tt.dot_scaled` element type (Triton's are E4M3, E5M2, E2M1, BF16 and
// FP16), so it must fail to compile. Formats used to be mapped by discriminant onto a list in a
// different order, which lowered Int8 as BF16 (and BF16x2 as fp4) without complaint.
fn dot_scaled_int8<T: Triton, D: Float>() {
    let ptrs = T::zeros::<T::Pointer<D>>(&[32, 32]);
    let a = T::load::<D, 2>(ptrs, None, None, &[0, 1], None, None, None, false);
    let scale = T::zeros::<i8>(&[32, 1]);
    let c = T::dot_scaled::<D, i8, D>(
        a,
        scale,
        DotFormat::Int8,
        a,
        scale,
        DotFormat::E4M3,
        None,
        false,
    );
    T::store::<D, 2>(ptrs, c, None, &[0, 1], None, None);
}

type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point() {
    dot_scaled_int8::<LlvmTriton, f32>();
}
