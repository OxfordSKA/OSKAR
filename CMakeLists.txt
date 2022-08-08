#
# Top level CMakeLists.txt
#

# === Set the project name and specify the required CMake version.
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
if (APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.9" CACHE STRING "Minimum macOS version")
endif()

# Set policies to avoid CMake warnings.
if (POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()
if (POLICY CMP0045)
    cmake_policy(SET CMP0045 NEW)
endif()
if (POLICY CMP0068)
    cmake_policy(SET CMP0068 NEW)
endif()
project(OSKAR CXX C)

# === Project version.
set(OSKAR_VERSION_ID "0x020804")
set(OSKAR_VERSION_MAJOR 2)
set(OSKAR_VERSION_MINOR 8)
set(OSKAR_VERSION_PATCH 4)
set(OSKAR_VERSION_SUFFIX "dev")

# === Add local CMake module path.
list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)
list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake/modules)

# === Get string length of current source directory for debug/log messages.
string(LENGTH "${CMAKE_SOURCE_DIR}/" SOURCE_PATH_SIZE)
add_definitions(-DSOURCE_PATH_SIZE=${SOURCE_PATH_SIZE})

# === Set install directory paths
if (NOT OSKAR_LIB_INSTALL_DIR)
    set(OSKAR_LIB_INSTALL_DIR "lib")
endif()
if (NOT OSKAR_INCLUDE_INSTALL_DIR)
    set(OSKAR_INCLUDE_INSTALL_DIR "include")
endif()
if (NOT OSKAR_BIN_INSTALL_DIR)
    set(OSKAR_BIN_INSTALL_DIR "bin")
endif()

# === Set compiler include directories.
include_directories(
    ${PROJECT_BINARY_DIR}
    ${PROJECT_BINARY_DIR}/oskar
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/oskar
    ${PROJECT_SOURCE_DIR}/extern
    ${PROJECT_SOURCE_DIR}/extern/gtest-1.7.0/include
    ${PROJECT_SOURCE_DIR}/extern/rapidxml-1.13
    ${PROJECT_SOURCE_DIR}/extern/cfitsio
    ${PROJECT_SOURCE_DIR}/extern/Random123
    ${PROJECT_SOURCE_DIR}/extern/Random123/features
)
set(GTEST_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/gtest-1.7.0/include/gtest)

# === Automatically set the build type if not specified.
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    # Use debug mode if building in dbg or debug directory.
    get_filename_component(dirname ${CMAKE_BINARY_DIR} NAME)
    if (${dirname} MATCHES "dbg" OR ${dirname} MATCHES "debug")
        set(CMAKE_BUILD_TYPE Debug)
    else()
        set(CMAKE_BUILD_TYPE Release)
    endif()
    message(STATUS "INFO: Setting CMAKE_BUILD_TYPE to ${CMAKE_BUILD_TYPE}")
endif()

# === Find dependencies and set definitions.
if (FIND_CUDA OR NOT DEFINED FIND_CUDA)
    find_package(CUDA 7.0)
endif()
if (FIND_OPENCL)
    if (NOT APPLE)
        set(OpenCL_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern)
    endif()
    if (MSVC)
        file(MAKE_DIRECTORY ${PROJECT_BINARY_DIR}/extern)
        file(TO_NATIVE_PATH ${PROJECT_SOURCE_DIR}/extern/CL/OpenCL.def def)
        file(TO_NATIVE_PATH ${PROJECT_BINARY_DIR}/extern/OpenCL.lib lib)
        execute_process(COMMAND lib /def:${def} /OUT:${lib})
        set(OpenCL_LIBRARY ${lib})
    endif()
    find_package(OpenCL QUIET)
endif()
find_package(OpenMP QUIET)
find_package(HDF5 QUIET)
find_package(Threads REQUIRED)
find_package(HARP QUIET)
if (CUDA_FOUND)
    add_definitions(-DOSKAR_HAVE_CUDA)
endif()
if (OpenCL_FOUND)
    add_definitions(-DOSKAR_HAVE_OPENCL)
    if (NOT APPLE)
        include_directories(${OpenCL_INCLUDE_DIRS})
    endif()
endif()
if (HDF5_FOUND)
    add_definitions(-DOSKAR_HAVE_HDF5)
    include_directories(${HDF5_INCLUDE_DIR})
endif()
if (HARP_FOUND)
    add_definitions(-DOSKAR_HAVE_HARP)
    include_directories(${HARP_INCLUDE_DIR})
endif()

# === Set compiler options.
include(oskar_set_version)
include(oskar_compiler_options)

# === Set up testing and code coverage.
if (BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
    enable_testing()
    if (CMAKE_BUILD_TYPE MATCHES Debug AND COVERAGE_REPORT)
        message(STATUS "INFO: Adding code coverage build target")
        include(CodeCoverage)
        set(COVERAGE_EXCLUDES
            "${PROJECT_BINARY_DIR}/*"
            "${PROJECT_SOURCE_DIR}/apps/*"
            "${PROJECT_SOURCE_DIR}/extern/*"
            "${PROJECT_SOURCE_DIR}/gui/*"
            "${PROJECT_SOURCE_DIR}/oskar/math/src/oskar_lapack_subset.c"
            "${PROJECT_SOURCE_DIR}/oskar/settings/extern/*"
            "${PROJECT_SOURCE_DIR}/oskar/splines/*"
            "${PROJECT_SOURCE_DIR}/.*/test/.*"
            "${PROJECT_SOURCE_DIR}/*/test/*"
            ".*.h"
            "*.h"
            "/usr/*"
            "/Applications/*"
            "/Users/*/Qt/*"
        )
        append_coverage_compiler_flags()
        setup_target_for_coverage_lcov(NAME coverage EXECUTABLE ctest)
    endif()
endif()

# === Build external libraries, OSKAR library and OSKAR applications.
add_subdirectory(extern)
add_subdirectory(oskar)
add_subdirectory(apps)
add_subdirectory(gui)

# === Build documentation.
add_subdirectory(docs)

# === Copy cleanup script into the build directory.
configure_file(${PROJECT_SOURCE_DIR}/cmake/oskar_cleanup.sh
    ${PROJECT_BINARY_DIR}/oskar_cleanup.sh @ONLY)

# === Include packaging script.
include(oskar_packaging)

# === Must be added last to run after all other install commands.
add_subdirectory(${PROJECT_SOURCE_DIR}/cmake/post_install)

