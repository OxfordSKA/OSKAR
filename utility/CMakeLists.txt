#
# src/utility/CMakeLists.txt
#

# ==== Utility library source
set(utility_SRC
    src/oskar_cuda_memory.cu
    src/oskar_load_csv_coordinates.cpp
)
set(utility_SRC "${utility_SRC}" PARENT_SCOPE)

# === Recurse into test directory.
add_subdirectory(test)
