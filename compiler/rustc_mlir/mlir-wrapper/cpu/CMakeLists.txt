# TritonCPU dialect, conversions and transforms, ported from the triton-cpu
# fork's third_party/cpu tree (see teenyc-3yu).
#
# The layout below mirrors upstream exactly -- cpu/{include,lib} -- because the
# ported sources spell their own headers as `#include "cpu/include/..."`. Those
# resolve against mlir-wrapper/ itself, which the parent CMakeLists.txt already
# puts on the include path for both the source and the (mirrored) build tree,
# so the tablegen'd .inc files land where the sources expect them with no
# extra include_directories() here.
#
# NOTE: this subtree does not compile against LLVM 22.0 yet and is therefore
# not built by default; see cpu/README.md for what remains. Enable with
# -DTRITON_CPU_ENABLE=ON once the port is finished.

# `#include "include/triton/Analysis/Utility.h"` in TritonToTritonCPU resolves
# against the Triton source root rather than its include/ dir, which the parent
# does not add.
if(DEFINED TRITON_SOURCE_DIR)
    include_directories(${TRITON_SOURCE_DIR})
endif()

add_subdirectory(include)
add_subdirectory(lib)
