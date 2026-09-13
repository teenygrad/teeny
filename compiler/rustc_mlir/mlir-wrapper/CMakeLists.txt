cmake_minimum_required(VERSION 3.20)
project(mlir-wrapper CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find LLVM
find_package(LLVM REQUIRED CONFIG)
message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Find MLIR
find_package(MLIR REQUIRED CONFIG)
message(STATUS "Using MLIRConfig.cmake in: ${MLIR_DIR}")

list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")

include(TableGen)
include(AddLLVM)
include(AddMLIR)
include(HandleLLVMOptions)

# Include directories
include_directories(${LLVM_INCLUDE_DIRS})
include_directories(${MLIR_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_SOURCE_DIR})
# Tablegen'd files (e.g. riscv/Dialect/RVV/IR/*.inc) land in the mirrored
# build-tree path; needed for our own dialects' #include "riscv/..." lines.
include_directories(${CMAKE_CURRENT_BINARY_DIR})

# Add Triton includes if available
if(DEFINED TRITON_SOURCE_DIR)
    message(STATUS "Triton source: ${TRITON_SOURCE_DIR}")
    include_directories(${TRITON_SOURCE_DIR}/include)
    include_directories(${TRITON_SOURCE_DIR}/third_party)
    include_directories(${TRITON_SOURCE_DIR}/lib)
    add_compile_definitions(TRITON_ENABLED=1)
endif()

if(DEFINED TRITON_BUILD_DIR)
    message(STATUS "Triton build: ${TRITON_BUILD_DIR}")
    include_directories(${TRITON_BUILD_DIR}/include)
    include_directories(${TRITON_BUILD_DIR}/third_party)
endif()

# Compiler flags
add_definitions(${LLVM_DEFINITIONS})

# Our own MLIR dialects (not Triton's -- see riscv/Dialect/RVV/IR/RVVDialect.td
# for why RVV lives here rather than in src/triton/third_party). Built as its
# own add_mlir_dialect_library() subproject (see riscv/CMakeLists.txt) rather
# than compiled directly into mlir-wrapper, so it's automatically picked up
# below via MLIR_DIALECT_LIBS instead of needing an explicit source/link entry.
add_subdirectory(riscv)

# TritonCPU dialect/conversions/transforms ported from the triton-cpu fork
# (teenyc-3yu). RiscvBackend lowers through these passes via CpuBackend, so the
# wrapper cannot be built without them; rustc_mlir/build.rs passes
# -DTRITON_CPU_ENABLE=ON.
option(TRITON_CPU_ENABLE "Build the ported TritonCPU dialect and passes" OFF)
if(NOT TRITON_CPU_ENABLE)
    message(FATAL_ERROR
        "mlir-wrapper requires -DTRITON_CPU_ENABLE=ON: RiscvBackend lowers "
        "through the TritonCPU passes (triton/backend/CpuBackend.cpp)")
endif()
add_subdirectory(cpu)

# Source files
set(WRAPPER_SOURCES
    MLIRWrapper.cpp
    Compiler.cpp
    triton/TritonWrapper.cpp
    triton/TritonCompiler.cpp
    triton/backend/Backend.cpp
    triton/backend/CudaBackend.cpp
    triton/backend/CpuBackend.cpp
    triton/backend/RiscvBackend.cpp
)

# Create static library
add_library(mlir-wrapper STATIC ${WRAPPER_SOURCES})

# CpuBackend.cpp includes the TritonCPU passes' tablegen'd headers, which the
# TritonCPU libraries generate, so those must be built first.
add_dependencies(mlir-wrapper
    TritonCPUIR
    TritonToTritonCPU
    TritonCPUTransforms
    TritonCPUToLLVM
)

# Get MLIR libraries
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)

# Link libraries
target_link_libraries(mlir-wrapper
    PRIVATE
    ${dialect_libs}
    ${extension_libs}
    MLIRIR
    MLIRSupport
    MLIRParser
    MLIRPass
    MLIRTransforms
    MLIRAnalysis
    LLVMSupport
    LLVMCore
    LLVMBitcodeReader
    LLVMLinker
    LLVMIRReader
    LLVMNVPTXCodeGen
    LLVMNVPTXDesc
    LLVMNVPTXInfo
)

# Link Triton if available
if(DEFINED TRITON_BUILD_DIR)
    target_link_directories(mlir-wrapper PRIVATE ${TRITON_BUILD_DIR}/lib)
    target_link_libraries(mlir-wrapper PRIVATE triton)
endif()

# Install
install(TARGETS mlir-wrapper
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
)
