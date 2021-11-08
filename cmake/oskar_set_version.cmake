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
    set(OSKAR_VERSION_SHORT "${OSKAR_VERSION}-${OSKAR_VERSION_SUFFIX}"
        CACHE STRING "Short OSKAR version"
    )
else()
    set(OSKAR_VERSION_SHORT "${OSKAR_VERSION}"
        CACHE STRING "Short OSKAR version"
    )
endif()

# Add the short Git hash for the long version string.
if (GIT_COMMIT_HASH)
    set(OSKAR_VERSION_LONG "${OSKAR_VERSION_SHORT}-${GIT_COMMIT_HASH}"
        CACHE STRING "Long OSKAR version"
    )
else()
    set(OSKAR_VERSION_LONG "${OSKAR_VERSION_SHORT}"
        CACHE STRING "Long OSKAR version"
    )
endif()

configure_file(${PROJECT_SOURCE_DIR}/cmake/oskar_version.h.in
    ${PROJECT_BINARY_DIR}/oskar/oskar_version.h @ONLY)
