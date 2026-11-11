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

// Regression test for: `codegen_cast`'s missing `CastKind::FloatToFloat` arm silently
// emitted `ub.poison` for scalar float widen/narrow casts (e.g. `x as f64`, `value as f32`),
// instead of a real `arith.extf`/`arith.truncf`. This is exactly the pattern
// `Float::from_f64(value: f64) -> Self` uses in vision-rs's generic Flash-Attention-2 /
// PSA kernels (`D::from_f64(x as f64)`), where the poisoned constant silently corrupted
// the online-softmax running max/scale, producing wrong (non-NaN, then eventually NaN)
// results with no compile error. See compiler/rustc_codegen_llvm/src/mlir/codegen/triton/mod.rs.
//
// Kernel: materialize a [BLOCK] tensor filled with `D::from_f64(x as f64)` (round-tripping
// the f32 kernel argument through f64, exactly like the generic Float trait does) and store
// it. With the bug, this compiled to `ub.poison`, register-allocated on top of whatever
// value happened to share that register — not caught by the compiler, only by comparing
// runtime output against the expected input.
fn float_to_float_scalar_cast<T: Triton, D: Float, const BLOCK: i32>(x: f32) {
    let out_ptrs = T::zeros::<T::Pointer<D>>(&[BLOCK]);
    let filled = T::full::<D>(&[BLOCK], D::from_f64(x as f64));
    T::store::<D, 1>(out_ptrs, filled, None, &[0], None, None);
}

use triton::llvm::triton::num::*;
use triton::llvm::triton::pointer::LlvmPointer;
type LlvmTriton = triton::llvm::triton::LlvmTriton;

#[no_mangle]
pub extern "C" fn entry_point(x: f32) {
    float_to_float_scalar_cast::<LlvmTriton, f32, 64>(x);
}
