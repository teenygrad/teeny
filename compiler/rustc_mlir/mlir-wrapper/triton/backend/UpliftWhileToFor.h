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

#pragma once

#include "mlir/IR/BuiltinOps.h"

namespace mlir {
namespace triton {

/// Uplifts every counted `scf.while` in `module` to `scf.for`.
///
/// This lives in its own translation unit, compiled without RTTI like LLVM
/// itself: the rewriter it uses has a vtable that references typeinfo LLVM
/// doesn't export.
void upliftWhileLoopsToFor(ModuleOp module);

} // namespace triton
} // namespace mlir
