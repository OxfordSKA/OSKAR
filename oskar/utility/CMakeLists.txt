#
# oskar/utility/CMakeLists.txt
#

set(utility_SRC
    oskar_kernel_macros.h
    oskar_vector_types_cl.h
    src/oskar_device_count.c
    src/oskar_device_create_list.cpp
    src/oskar_device_get_info.c
    src/oskar_device_log.c
    src/oskar_device.cpp
    src/oskar_dir.c
    src/oskar_file_exists.c
    src/oskar_get_binary_tag_string.c
    src/oskar_get_error_string.c
    src/oskar_get_memory_usage.c
    src/oskar_get_num_procs.c
    src/oskar_getline.c
    src/oskar_hdf5.c
    src/oskar_lock_file.c
    src/oskar_thread.c
    src/oskar_string_to_array.c
    src/oskar_timer.c
    src/oskar_version_string.c
)

set(utility_SRC "${utility_SRC}" PARENT_SCOPE)

# === Recurse into test directory.
add_subdirectory(test)
