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

// teenyc-32u: make_tensor_descriptor / load_tensor_descriptor emitted ub.poison when a slice
// argument could not be resolved, e.g. a literal slice that MIR promotes to a constant. The
// descriptor is copied tile by tile from `src` to `dst` with literal shapes and offsets.
fn copy_tile<T: Triton, D: Float>(src: T::Pointer<D>, dst: T::Pointer<D>) {
    let src_desc = T::make_tensor_descriptor::<D>(src, &[64, 64], &[64, 1], &[16, 16], None);
    let dst_desc = T::make_tensor_descriptor::<D>(dst, &[64, 64], &[64, 1], &[16, 16], None);
    let tile = T::load_tensor_descriptor::<D>(src_desc, &[16, 32]);
    T::store_tensor_descriptor::<D>(dst_desc, &[16, 32], tile);
}

use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(src: *mut f32, dst: *mut f32) {
    copy_tile::<LlvmTriton, f32>(LlvmPointer(src as *mut _), LlvmPointer(dst as *mut _));
}
