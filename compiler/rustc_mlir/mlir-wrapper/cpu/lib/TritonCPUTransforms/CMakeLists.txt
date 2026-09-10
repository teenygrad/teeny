# See lib/Dialect/TritonCPU/IR/CMakeLists.txt for why this is
# add_mlir_dialect_library() rather than upstream's add_triton_library().
add_mlir_dialect_library(TritonCPUTransforms
    ConvertDotOp/ConvertDotCommon.cpp
    ConvertDotOp/ConvertDotGeneric.cpp
    ConvertDotOp/ConvertDotToAMX.cpp
    ConvertDotOp/ConvertDotToFMA.cpp
    ConvertDotOp/ConvertDotOpToUkernelOps.cpp
    ConvertDotOp/ConvertDotToNanokernel.cpp
    Canonicalize.cpp
    ConvertDotProduct.cpp
    ConvertUnsupportedOps.cpp
    DecomposeFpConversions.cpp
    OptimizeMasks.cpp
    UnrollAndReorderElementwiseOps.cpp

    DEPENDS
    TritonCPUTransformsPassIncGen

    LINK_LIBS PUBLIC
    TritonCPUIR
)

target_compile_options(TritonCPUTransforms PRIVATE -fno-rtti)

# Carried over from upstream: the OneDNN ukernel path links libdnnl when it is
# present. Kept as an optional dependency rather than a hard one.
if (dnnl_FOUND)
  target_link_libraries(TritonCPUTransforms PRIVATE DNNL::dnnl)
endif()
