set(OSKAR_VERSION "${OSKAR_VERSION_MAJOR}.${OSKAR_VERSION_MINOR}.${OSKAR_VERSION_PATCH}")
set(OSKAR_VERSION_STR "${OSKAR_VERSION}")
if (OSKAR_VERSION_SUFFIX AND NOT OSKAR_VERSION_SUFFIX STREQUAL "")
    find_package(Git QUIET)
    if (GIT_FOUND)
        execute_process(
          COMMAND git log -1 --format=%h
          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
          OUTPUT_VARIABLE GIT_COMMIT_HASH
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        execute_process(
          COMMAND git log -1 --format=%cd --date=short
          WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
          OUTPUT_VARIABLE GIT_COMMIT_DATE
          OUTPUT_STRIP_TRAILING_WHITESPACE
        )
        set(GIT_COMMIT_INFO "${GIT_COMMIT_DATE} ${GIT_COMMIT_HASH}")
    endif()
    set(OSKAR_VERSION_STR "${OSKAR_VERSION}-${OSKAR_VERSION_SUFFIX}")
    if (GIT_COMMIT_INFO)
        set(OSKAR_VERSION_STR "${OSKAR_VERSION_STR} ${GIT_COMMIT_INFO}")
    endif()
    if (CMAKE_BUILD_TYPE MATCHES [dD]ebug)
        set(OSKAR_VERSION_STR "${OSKAR_VERSION_STR} -debug-")
    endif()
endif()

configure_file(${PROJECT_SOURCE_DIR}/cmake/oskar_version.h.in
    ${PROJECT_BINARY_DIR}/oskar/oskar_version.h @ONLY)