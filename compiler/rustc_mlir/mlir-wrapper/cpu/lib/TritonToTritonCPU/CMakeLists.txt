# See lib/Dialect/TritonCPU/IR/CMakeLists.txt for why this is
# add_mlir_dialect_library() rather than upstream's add_triton_library().
add_mlir_dialect_library(TritonToTritonCPU
    ConvertAtomicOps.cpp
    ConvertControlFlowOps.cpp
    ConvertDebugOps.cpp
    ConvertDotOp.cpp
    ConvertElementwiseOps.cpp
    ConvertElemManipOps.cpp
    ConvertHistogramOp.cpp
    ScalarizeInterface.cpp
    ScalarizeUsingForOps.cpp
    ConvertMemoryOps.cpp
    ConvertPtrOps.cpp
    ConvertReductionOp.cpp
    ConvertScanOp.cpp
    TypeConverter.cpp

    DEPENDS
    TritonToTritonCPUPassIncGen
    ScalarizeInterfaceIncGen
    MLIRDialectUtils

    LINK_LIBS PUBLIC
    TritonCPUIR
    MLIRVectorDialect
)

target_compile_options(TritonToTritonCPU PRIVATE -fno-rtti)
