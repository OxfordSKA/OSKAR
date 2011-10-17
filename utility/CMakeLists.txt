#
# src/utility/CMakeLists.txt
#

# ==== Utility library source
set(utility_SRC
    src/oskar_cuda_device_info.cu
    src/oskar_cuda_memory.cu
    src/oskar_load_csv_coordinates_2d.c
    src/oskar_load_csv_coordinates_3d.c
    src/oskar_mem_alloc.cpp
    src/oskar_mem_copy.cpp
    src/oskar_mem_element_size.c
    src/oskar_mem_free.cpp
    src/oskar_mem_realloc.c
    src/oskar_Mem.cpp
)
set(utility_SRC "${utility_SRC}" PARENT_SCOPE)

add_subdirectory(matlab)

# === Recurse into test directory.
add_subdirectory(test)
