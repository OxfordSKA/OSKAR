# CMake initialisation.
cmake_minimum_required(VERSION 3.1)
if (POLICY CMP0045)
    cmake_policy(SET CMP0045 NEW)
endif()
if (POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
endif()

# Declare the CMake project.
project(oskar_binary C)

# Set the OSKAR binary version used to version dynamic libraries.
set(OSKAR_BINARY_VERSION "2.1")

# Set the name of the library.
set(libname ${PROJECT_NAME})

# If OSKAR_VERSION_STR is not defined, we are building as stand-alone library.
if (NOT OSKAR_VERSION_STR)
    set(STANDALONE_BUILD ON)
endif()
add_definitions(-D_FILE_OFFSET_BITS=64)
if (NOT WIN32)
    add_definitions(-D_POSIX_C_SOURCE=200808L)
endif()

# Macros for building oskar_binary as a standalone library.
if (STANDALONE_BUILD)

    # Build type detection.
    if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
        get_filename_component(dirname ${CMAKE_BINARY_DIR} NAME)
        if (${dirname} MATCHES "dbg" OR ${dirname} MATCHES "debug")
            set(CMAKE_BUILD_TYPE Debug)
        else()
            set(CMAKE_BUILD_TYPE Release)
        endif()
    endif()

    # Enable the ctest framework.
    enable_testing()

    # Compiler options for standalone mode.
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    if ("${CMAKE_C_COMPILER_ID}" MATCHES ".*Clang.*" OR "${CMAKE_C_COMPILER_ID}" STREQUAL "GNU")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c99")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fvisibility=hidden")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fdiagnostics-show-option")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -Wextra")
        set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -pedantic")
    endif()
    include_directories(${PROJECT_SOURCE_DIR})

    # Set install locations for standalone mode.
    if (NOT OSKAR_LIB_INSTALL_DIR)
        set(OSKAR_LIB_INSTALL_DIR "lib")
    endif()
    if (NOT OSKAR_INCLUDE_INSTALL_DIR)
        set(OSKAR_INCLUDE_INSTALL_DIR "include/oskar/binary")
    endif()

endif(STANDALONE_BUILD)

# Declare build and install targets for the library.
file(GLOB ${libname}_SRC src/*.c)
add_library(${libname} ${${libname}_SRC})
set_target_properties(${libname} PROPERTIES
    SOVERSION ${OSKAR_BINARY_VERSION}
    VERSION ${OSKAR_BINARY_VERSION})
install(TARGETS ${libname}
    ARCHIVE DESTINATION ${OSKAR_LIB_INSTALL_DIR} COMPONENT libraries
    LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR} COMPONENT libraries
    RUNTIME DESTINATION ${OSKAR_BIN_INSTALL_DIR} COMPONENT runtime
)

# Add unit tests / examples.
add_subdirectory(test)

# Install headers and packaging macros.
if (STANDALONE_BUILD)
    file(GLOB ${libname}_H oskar*.h) # Public headers only (start with oskar_)
    install(FILES ${${libname}_H} DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR})

    set(CPACK_PACKAGE_NAME "OSKAR_binary")
    set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/readme.txt")
    set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OSKAR binary library (version ${OSKAR_BINARY_VERSION})")
    set(CPACK_PACKAGE_VENDOR "University of Oxford, Oxford e-Research Centre.")
    set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
    set(CPACK_PACKAGE_VERSION_MAJOR "${OSKAR_BINARY_VERSION}")
    set(CPACK_PACKAGE_VERSION "${OSKAR_BINARY_VERSION}")
    set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)
    set(CPACK_GENERATOR "TGZ")
    set(CPACK_SOURCE_GENERATOR "ZIP")
    set(CPACK_SOURCE_IGNORE_FILES "/build/;/debug/;/dbg/;/release/")
    include(CPack)
    add_custom_target(dist
        COMMAND ${CMAKE_MAKE_PROGRAM} package_source
        COMMENT "Packaging Source files"
        VERBATIM)
endif(STANDALONE_BUILD)
