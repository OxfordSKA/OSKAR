#
# src/utility/CMakeLists.txt
#

# ==== Utility library source
set(utility_SRC
    src/oskar_blank_parentheses.c
    src/oskar_cuda_device_info.cu
    src/oskar_cuda_memory.cu
    src/oskar_exit.c
    src/oskar_get_error_string.c
    src/oskar_getline.c
    src/oskar_mem_add_real_gaussian_noise.c
    src/oskar_mem_add_gaussian_noise.c
    src/oskar_mem_add.c
    src/oskar_mem_alloc.c
    src/oskar_mem_append_raw.c
    src/oskar_mem_append.c
    src/oskar_mem_clear_contents.c
    src/oskar_mem_copy.c
    src/oskar_mem_element_size.c
    src/oskar_mem_free.c
    src/oskar_mem_get_pointer.c
    src/oskar_mem_init.c
    src/oskar_mem_insert.c
    src/oskar_mem_realloc.c
    src/oskar_mem_scale_real.cu
    src/oskar_Mem.cpp
    src/oskar_string_to_array.c
    src/oskar_work_init.c
    src/oskar_Work.cpp
)
set(utility_SRC "${utility_SRC}" PARENT_SCOPE)

add_subdirectory(matlab)

# === Recurse into test directory.
add_subdirectory(test)
