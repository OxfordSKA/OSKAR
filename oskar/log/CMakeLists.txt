#
# oskar/log/CMakeLists.txt
#

set(log_SRC
    src/oskar_log_accessors.c
    src/oskar_log_create.c
    src/oskar_log_error.c
    src/oskar_log_file_data.c
    src/oskar_log_file_exists.c
    src/oskar_log_free.c
    src/oskar_log_line.c
    src/oskar_log_message.c
    src/oskar_log_section.c
    src/oskar_log_system_clock_data.c
    src/oskar_log_system_clock_string.c
    src/oskar_log_value.c
    src/oskar_log_warning.c
    src/oskar_log_write.c
)

set(log_SRC "${log_SRC}" PARENT_SCOPE)

add_subdirectory(test)
