#
# Top level CMakeLists.txt
#

# === Set the project name and specify the required CMAKE version.
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
if (APPLE)
    set(CMAKE_OSX_DEPLOYMENT_TARGET "10.8" CACHE STRING "Minimum macOS version")
endif()

# Set various policies to get old behviour with cmake 3.x
if (POLICY CMP0042)
    cmake_policy(SET CMP0042 OLD)
endif()
if (POLICY CMP0045)
    cmake_policy(SET CMP0045 OLD)
endif()
project(OSKAR CXX C)

# === Project version.
set(OSKAR_VERSION_ID "0x020700")
set(OSKAR_VERSION_MAJOR 2)
set(OSKAR_VERSION_MINOR 7)
set(OSKAR_VERSION_PATCH 0)
set(OSKAR_VERSION_SUFFIX "RC")

# === Add local CMake module path.
list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)
list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake/modules)

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
    ${PROJECT_SOURCE_DIR}/extern/cfitsio-3.41
    ${PROJECT_SOURCE_DIR}/extern/Random123
    ${PROJECT_SOURCE_DIR}/extern/Random123/features
)
set(GTEST_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/extern/gtest-1.7.0/include/gtest)

# === Set dependencies and compiler options.
include(oskar_dependencies)
include(oskar_set_version)
include(oskar_compiler_options)
enable_testing()

# === Build external libraries, OSKAR library and OSKAR applications.
add_subdirectory(extern)
add_subdirectory(oskar)
add_subdirectory(apps)
add_subdirectory(gui)

# === Build documentation.
add_subdirectory(doc)

# === Copy cleanup script into the build directory.
configure_file(${PROJECT_SOURCE_DIR}/cmake/oskar_cleanup.sh
    ${PROJECT_BINARY_DIR}/oskar_cleanup.sh @ONLY)

# === Include packaging script.
include(oskar_packaging)
