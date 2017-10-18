#
# oskar/utility/CMakeLists.txt
#

set(utility_SRC
    src/oskar_binary_write_metadata.c
    src/oskar_cl_utils.cpp
    src/oskar_device_utils.c
    src/oskar_dir.c
    src/oskar_file_exists.c
    src/oskar_get_error_string.c
    src/oskar_get_memory_usage.c
    src/oskar_get_num_procs.c
    src/oskar_getline.c
    src/oskar_thread.c
    src/oskar_scan_binary_file.c
    src/oskar_string_to_array.c
    src/oskar_timer.c
    src/oskar_version_string.c
)

if (CUDA_FOUND)
    list(APPEND utility_SRC
        src/oskar_cuda_info_create.c
        src/oskar_cuda_info_free.c
        src/oskar_cuda_info_log.c
        src/oskar_cuda_mem_log.c
    )
endif()

set(utility_SRC "${utility_SRC}" PARENT_SCOPE)

# === Recurse into test directory.
add_subdirectory(test)

