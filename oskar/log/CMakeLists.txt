#
# oskar/log/CMakeLists.txt
#

set(log_SRC
    src/oskar_log_create.c
    src/oskar_log_file_data.c
    src/oskar_log_free.c
    src/oskar_log.c
)

set(log_SRC "${log_SRC}" PARENT_SCOPE)

add_subdirectory(test)
