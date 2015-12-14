#
# src/CMakeLists.txt (top level CMakeLists.txt)
#

# === Set the project name and specify the required CMAKE version.
cmake_minimum_required(VERSION 2.8.3 FATAL_ERROR)
# Set various policies to get old behviour with cmake 3.x
if (POLICY CMP0042)
    cmake_policy(SET CMP0042 OLD)
endif()
if (POLICY CMP0045)
    cmake_policy(SET CMP0045 OLD)
endif()
project(OSKAR CXX C)

# === Project version.
set(OSKAR_VERSION_ID "0x020601")
set(OSKAR_VERSION_MAJOR 2)
set(OSKAR_VERSION_MINOR 6)
set(OSKAR_VERSION_PATCH 1)
set(OSKAR_VERSION_SUFFIX "")

# === Add a local module path
list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake)
list(INSERT CMAKE_MODULE_PATH 0 ${PROJECT_SOURCE_DIR}/cmake/modules)

# === Setup install directory paths
if (NOT OSKAR_LIB_INSTALL_DIR)
    set(OSKAR_LIB_INSTALL_DIR "lib")
endif()
if (NOT OSKAR_INCLUDE_INSTALL_DIR)
    set(OSKAR_INCLUDE_INSTALL_DIR "include/oskar")
endif()
if (NOT OSKAR_BIN_INSTALL_DIR)
    set(OSKAR_BIN_INSTALL_DIR "bin")
endif()
#if (NOT OSKAR_MATLAB_INSTALL_DIR)
#    set(OSKAR_MATLAB_INSTALL_DIR "share/oskar/matlab")
#endif()
#if (NOT OSKAR_PYTHON_INSTALL_DIR)
#    set(OSKAR_PYTHON_INSTALL_DIR "share/oskar/python")
#endif()

# === Set dependencies and compiler options.
include(oskar_dependencies)
include(oskar_compiler_options)

enable_testing()

# === Build main OSKAR library.
if (CUDA_FOUND)
    set(libname oskar)
    # List of source subfolder modules to be build into the oskar library.
    set(${libname}_modules
        correlate
        element
        convert
        imaging
        interferometry
        jones
        math
        sky
        splines
        station
        utility)
    foreach (module ${${libname}_modules})
        add_subdirectory(${module})
        foreach (file ${${module}_SRC})
            list(APPEND ${libname}_SRC ${module}/${file})
        endforeach()
    endforeach()

    set(CUDA_GENERATED_OUTPUT_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${libname}.dir/src)

    cuda_add_library(${libname} ${${libname}_SRC})
    target_link_libraries(${libname}
        oskar_settings_struct
        oskar_log
        oskar_binary
        ${CUDA_curand_LIBRARY})

    if (OSKAR_USE_LAPACK)
        target_link_libraries(${libname} ${OSKAR_LAPACK})
    endif ()
    if (MSVC)
        set_target_properties(${libname} PROPERTIES
            COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
            SOVERSION ${OSKAR_VERSION})
    else ()
        set_target_properties(${libname} PROPERTIES
            INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
            INSTALL_RPATH_USE_LINK_PATH TRUE
            COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
            LINK_FLAGS "${OpenMP_CXX_FLAGS}"
            SOVERSION ${OSKAR_VERSION}
            VERSION ${OSKAR_VERSION})
    endif()

    install(TARGETS ${libname} DESTINATION ${OSKAR_LIB_INSTALL_DIR})

    cuda_build_clean_target()
endif ()

# === Build other oskar libraries.
add_subdirectory(apps)
add_subdirectory(fits)
add_subdirectory(ms)
add_subdirectory(settings)
#add_subdirectory(matlab)
#add_subdirectory(python)

# === Build external libraries.
add_subdirectory(extern)

# === Build documentation.
add_subdirectory(doc)

# === Install header tree.
if (NOT ${CMAKE_INSTALL_PREFIX} MATCHES "/usr/local")
    string(REGEX MATCH "[^/|\\][a-zA-z0-9|_|-]+$" bin_dir ${CMAKE_BINARY_DIR})
    string(REGEX MATCH "[^/|\\]([a-zA-z0-9|_|-]+$)" install_dir
        ${CMAKE_INSTALL_PREFIX})
    string(COMPARE EQUAL ${CMAKE_INSTALL_PREFIX} ${CMAKE_BINARY_DIR}
        install_in_build_dir)
    if ($install_in_build_dir)
        set(install_include_dir "${bin_dir}/include")
    else ()
        set(install_include_dir "${bin_dir}/${install_dir}/include")
    endif()
else()
    set(install_include_dir "${CMAKE_INSTALL_PREFIX}/include")
endif ()
# TODO move install command into subdirectories to make this more modular,
# and give subdirectories control over which headers are installed.
install(DIRECTORY "${PROJECT_SOURCE_DIR}/."
    DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
    FILES_MATCHING PATTERN "oskar_*.h"
    PATTERN ${install_include_dir} EXCLUDE
    PATTERN build EXCLUDE
    PATTERN release EXCLUDE
    PATTERN release* EXCLUDE
    PATTERN debug EXCLUDE
    PATTERN debug* EXCLUDE
    PATTERN .settings EXCLUDE
    PATTERN cmake EXCLUDE
    PATTERN doc EXCLUDE
    PATTERN gui EXCLUDE
    PATTERN matlab EXCLUDE
    PATTERN python EXCLUDE
    PATTERN *dierckx* EXCLUDE
    PATTERN src  EXCLUDE
    PATTERN test EXCLUDE
    PATTERN extern EXCLUDE
    PATTERN .svn EXCLUDE)
install(FILES ${PROJECT_BINARY_DIR}/oskar_version.h
    DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR})

# === Copy cleanup bash script into the build directory
configure_file(${PROJECT_SOURCE_DIR}/cmake/oskar_cleanup.sh
    ${PROJECT_BINARY_DIR}/oskar_cleanup.sh @ONLY)

# === Include packaging script.
include(oskar_packaging)
