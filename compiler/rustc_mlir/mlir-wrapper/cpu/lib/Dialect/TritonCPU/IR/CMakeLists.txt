# add_triton_library() upstream; add_mlir_dialect_library() here -- the former
# is a macro from Triton's own CMake and is not available in this build. The
# MLIR form also registers the target in the global MLIR_DIALECT_LIBS property,
# which mlir-wrapper already links, so no explicit entry is needed there.
add_mlir_dialect_library(TritonCPUIR
    Dialect.cpp
    Ops.cpp
    Types.cpp

    DEPENDS
    TritonCPUTableGen
    TritonCPUAttrDefsIncGen
)

# Upstream has `LINK_LIBS PUBLIC TritonIR`. That is a CMake target inside
# Triton's own build tree and is not visible from this standalone project --
# Triton reaches us as a single prebuilt libtriton.a, not as per-dialect
# targets. Dropping it is safe: these are static libraries, so Triton's symbols
# only need to resolve when the final consumer links, and mlir-wrapper already
# pulls in libtriton there.

# LLVM/MLIR are built with RTTI disabled, so typeinfo for their polymorphic
# bases (mlir::Dialect, mlir::Type, ...) is never emitted. Defining classes
# derived from them with RTTI on fails to link; see the longer note in
# riscv/Dialect/RVV/IR/CMakeLists.txt.
target_compile_options(TritonCPUIR PRIVATE -fno-rtti)
