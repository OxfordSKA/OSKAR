#
# oskar/log/CMakeLists.txt
#

set(log_SRC
    src/oskar_log.c
)

set(log_SRC "${log_SRC}" PARENT_SCOPE)

if (BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
    add_subdirectory(test)
endif()
