#
# src/utility/CMakeLists.txt
#

set(utility_SRC
    src/oskar_binary_create.c
    src/oskar_binary_free.c
    src/oskar_binary_query.c
    src/oskar_binary_read_mem.c
    src/oskar_binary_read_oskar_version.c
    src/oskar_binary_read.c
    src/oskar_binary_write_mem.c
    src/oskar_binary_write_metadata.c
    src/oskar_binary_write.c
    src/oskar_blank_parentheses.c
    src/oskar_endian.c
    src/oskar_file_exists.c
    src/oskar_get_error_string.c
    src/oskar_getline.c
    src/oskar_mem_accessors.c
    src/oskar_mem_add.c
    src/oskar_mem_alloc.c
    src/oskar_mem_append_raw.c
    src/oskar_mem_append.c
    src/oskar_mem_clear_contents.c
    src/oskar_mem_convert_precision.c
    src/oskar_mem_copy.c
    src/oskar_mem_create_alias_from_raw.c
    src/oskar_mem_create_alias.c
    src/oskar_mem_create_copy.c
    src/oskar_mem_create.c
    src/oskar_mem_data_type_string.c
    src/oskar_mem_different.c
    src/oskar_mem_element_size.c
    src/oskar_mem_element_multiply.c
    src/oskar_mem_evaluate_relative_error.c
    src/oskar_mem_free.c
    src/oskar_mem_get_element.c
    src/oskar_mem_insert.c
    src/oskar_mem_load_ascii.c
    src/oskar_mem_random_fill.c
    src/oskar_mem_read_binary_raw.c
    src/oskar_mem_realloc.c
    src/oskar_mem_save_ascii.c
    src/oskar_mem_scale_real.c
    src/oskar_mem_set_alias.c
    src/oskar_mem_set_element.c
    src/oskar_mem_set_value_real.c
    src/oskar_mem_stats.c
    src/oskar_scan_binary_file.c
    src/oskar_string_to_array.c
    src/oskar_system_clock_data.c
    src/oskar_system_clock_string.c
    src/oskar_timer.c
    src/oskar_timers.c
    src/oskar_version_string.c
)

# TODO move this into the apps folder! (log should ONLY be used for apps)
list(APPEND utility_SRC
    log/src/oskar_log_accessors.c
    log/src/oskar_log_create.c
    log/src/oskar_log_error.c
    log/src/oskar_log_file_data.c
    log/src/oskar_log_free.c
    log/src/oskar_log_line.c
    log/src/oskar_log_message.c
    log/src/oskar_log_section.c
    log/src/oskar_log_value.c
    log/src/oskar_log_warning.c
    log/src/oskar_log_write.c
)

if (CUDA_FOUND)
    list(APPEND utility_SRC
        src/oskar_cuda_check_error.c
        src/oskar_cuda_device_info_scan.c
        src/oskar_cuda_info_create.c
        src/oskar_cuda_info_free.c
        src/oskar_cuda_info_log.c
        src/oskar_cuda_mem_log.c
        src/oskar_mem_element_multiply_cuda.cu
        src/oskar_mem_scale_real_cuda.cu
        src/oskar_mem_set_value_real_cuda.cu
        src/oskar_random_state_create.cu
        src/oskar_random_state_free.c
    )
endif()

set(utility_SRC "${utility_SRC}" PARENT_SCOPE)

# === Recurse into test directory.
add_subdirectory(test)
