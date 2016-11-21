#
# Top level CMakeLists.txt
#

# === Set the project name and specify the required CMAKE version.
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
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
set(OSKAR_VERSION_SUFFIX "dev")

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
if (NOT OSKAR_PYTHON_INSTALL_DIR)
    set(OSKAR_PYTHON_INSTALL_DIR "share/oskar/python")
endif()

# === Set dependencies and compiler options.
include(oskar_dependencies)
include(oskar_includes)
include(oskar_set_version)
include(oskar_compiler_options)
enable_testing()

# === Build external libraries, OSKAR library and OSKAR applications.
add_subdirectory(extern)
add_subdirectory(oskar)
add_subdirectory(apps)
add_subdirectory(python)

# === Build documentation.
add_subdirectory(doc)

# === Copy cleanup script into the build directory.
configure_file(${PROJECT_SOURCE_DIR}/cmake/oskar_cleanup.sh
    ${PROJECT_BINARY_DIR}/oskar_cleanup.sh @ONLY)

# === Include packaging script.
include(oskar_packaging)
