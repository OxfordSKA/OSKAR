# CMake initialisation.
cmake_minimum_required(VERSION 3.1)
if (POLICY CMP0045)
    cmake_policy(SET CMP0045 NEW)
endif()
if (POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
endif()

# Declare the CMake project.
project(oskar_beam_utils CXX C)

# Set the version used for dynamic libraries.
set(OSKAR_BEAM_UTILS_VERSION "1")

# Set the name of the library.
set(libname ${PROJECT_NAME})

# If OSKAR_VERSION_STR is not defined, we are building a stand-alone library.
if (NOT OSKAR_VERSION_STR)
    set(STANDALONE_BUILD ON)
endif()

# Macros for building oskar_beam_utils as a standalone library.
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

    # Compiler options for standalone mode.
    set(BUILD_SHARED_LIBS ON)
    set(CMAKE_POSITION_INDEPENDENT_CODE ON)
    include_directories(
        "${PROJECT_SOURCE_DIR}/"
        "${PROJECT_SOURCE_DIR}/../"
    )

    # Set install locations for standalone mode.
    if (NOT OSKAR_LIB_INSTALL_DIR)
        set(OSKAR_LIB_INSTALL_DIR "lib")
    endif()
    if (NOT OSKAR_INCLUDE_INSTALL_DIR)
        set(OSKAR_INCLUDE_INSTALL_DIR "include/oskar/beam_utils")
    endif()

endif(STANDALONE_BUILD)

# Declare build and install targets for the library.
set(${libname}_SRC
    src/oskar_beam_utils.cpp
)
add_library(${libname} ${${libname}_SRC})
set_target_properties(${libname} PROPERTIES
    SOVERSION ${OSKAR_BEAM_UTILS_VERSION}
    VERSION ${OSKAR_BEAM_UTILS_VERSION})
install(TARGETS ${libname}
    ARCHIVE DESTINATION ${OSKAR_LIB_INSTALL_DIR} COMPONENT libraries
    LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR} COMPONENT libraries
    RUNTIME DESTINATION ${OSKAR_BIN_INSTALL_DIR} COMPONENT runtime
)

# Install headers.
if (STANDALONE_BUILD)
    file(GLOB ${libname}_H oskar*.h) # Public headers only.
    install(FILES ${${libname}_H} DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR})
endif(STANDALONE_BUILD)
