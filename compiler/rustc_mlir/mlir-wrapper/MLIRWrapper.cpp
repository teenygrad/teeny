//===-- MLIRWrapper.cpp - C bindings for MLIR types -----------------------===//
//
// Implementation of C-compatible bindings for MLIR types.
//
//===----------------------------------------------------------------------===//

#include "MLIRWrapper.h"

#include "mlir/IR/DialectRegistry.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;

extern "C" void mlirLoadTritonDialect(MlirContext ctx) {
  MLIRContext *context = unwrap(ctx);
  DialectRegistry registry;

  registry.insert<triton::TritonDialect>();
  context->appendDialectRegistry(registry);

  context->loadDialect<triton::TritonDialect>();
}

extern "C" MlirType mlirCreateTritonPointerType(MlirType pointee,
                                                int address_space) {
  auto type = unwrap(pointee);

  // The FFI keeps a plain int; Triton 3.9 made the parameter a PtrAddrSpace
  // enum whose values are the address-space numbers themselves.
  auto pointer_type = triton::getPointerType(
      type, static_cast<triton::PtrAddrSpace>(address_space));

  return wrap(pointer_type);
}
