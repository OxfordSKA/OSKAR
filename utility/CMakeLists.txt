#
# src/utility/CMakeLists.txt
#

# ==== Needed?
include_directories(${oskar-lib_SOURCE_DIR}/utility)

# ==== Utility library source
set(utility_SRC
    src/oskar_util_cuda_memory.cu
)
set(utility_SRC "${utility_SRC}" PARENT_SCOPE)

# === Recurse into test directory.
add_subdirectory(test)
