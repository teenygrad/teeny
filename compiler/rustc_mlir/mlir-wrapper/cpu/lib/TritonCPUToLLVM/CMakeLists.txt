# See lib/Dialect/TritonCPU/IR/CMakeLists.txt for why this is
# add_mlir_dialect_library() rather than upstream's add_triton_library().
#
# The two Ukernel* sources are kept in the build deliberately: Passes.td
# registers createConvertDotOpToUkernelOps()/the ukernel lowerings
# unconditionally, so dropping the .cpp files leaves registerTritonCPUToLLVM-
# Passes() with undefined references. Trimming them means trimming Passes.td
# too -- part of the port, not something to paper over here.
add_mlir_dialect_library(TritonCPUToLLVM
    AtomicOpsToLLVM.cpp
    DebugOpsToLLVM.cpp
    UkernelOpsToOneDNNLLVM.cpp
    UkernelOpsToXSMMLLVM.cpp
    FuncOpToLLVM.cpp
    GetProgramIdOpToLLVM.cpp
    LowerMultiReduction.cpp
    MathToVecLib.cpp
    MemoryOpToLLVM.cpp
    TypeConverter.cpp
    Utility.cpp

    DEPENDS
    TritonCPUToLLVMConversionPassIncGen

    LINK_LIBS PUBLIC
    TritonCPUIR
    MLIRVectorToLLVMPass
)

target_compile_options(TritonCPUToLLVM PRIVATE -fno-rtti)
