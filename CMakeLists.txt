#
# src/CMakeLists.txt (top level CMakeLists.txt)
#

# HACK FOR OS X to prevent the C++ compiler using clang.
# This is because clang doesnt have OpenMP support.
if (APPLE AND NOT CMAKE_CXX_COMPILER)
    set(CMAKE_CXX_COMPILER g++)
    set(CMAKE_C_COMPILER gcc)
endif()

if (DEFINED ENV{CMAKE_CXX_COMPILER})
    set(CMAKE_CXX_COMPILER $ENV{CMAKE_CXX_COMPILER})
endif()

if (DEFINED ENV{CMAKE_C_COMPILER})
    set(CMAKE_C_COMPILER $ENV{CMAKE_C_COMPILER})
endif()

# === Set the project name and specify the required CMAKE version.
project(OSKAR CXX C)
cmake_minimum_required(VERSION 2.6 FATAL_ERROR)

# === Project version.
set(OSKAR_VERSION_MAJOR 2)
set(OSKAR_VERSION_MINOR 5)
set(OSKAR_VERSION_PATCH 0)
set(OSKAR_VERSION "${OSKAR_VERSION_MAJOR}.${OSKAR_VERSION_MINOR}.${OSKAR_VERSION_PATCH}")

# === Setup install directory paths
# Note: the install directory prefix is set by -DCMAKE_INSTALL_PRFEIX=<path>
if (NOT OSKAR_LIB_INSTALL_DIR)
    set(OSKAR_LIB_INSTALL_DIR "lib")
endif()
if (NOT OSKAR_INCLUDE_INSTALL_DIR)
    set(OSKAR_INCLUDE_INSTALL_DIR "include/oskar")
endif()
if (NOT OSKAR_BIN_INSTALL_DIR)
    set(OSKAR_BIN_INSTALL_DIR "bin")
endif()
if (NOT OSKAR_MATLAB_INSTALL_DIR)
    set(OSKAR_MATLAB_INSTALL_DIR "share/oskar/matlab")
endif()
if (NOT OSKAR_PYTHON_INSTALL_DIR)
    set(OSKAR_PYTHON_INSTALL_DIR "share/oskar/python")
endif()


# === Set dependencies and compiler options.
include(cmake/oskar_dependencies.cmake)
include(cmake/oskar_compiler_options.cmake)

enable_testing()

# === Build main OSKAR library.
if (CUDA_FOUND)

    add_subdirectory(imaging)
    add_subdirectory(interferometry)
    add_subdirectory(math)
    add_subdirectory(measures)
    add_subdirectory(sky)
    add_subdirectory(station)
    add_subdirectory(utility)

    set(libname oskar)
    foreach (file ${measures_SRC})
        list(APPEND ${libname}_SRC measures/${file})
    endforeach ()
    foreach (file ${imaging_SRC})
        list(APPEND ${libname}_SRC imaging/${file})
    endforeach ()
    foreach (file ${interferometry_SRC})
        list(APPEND ${libname}_SRC interferometry/${file})
    endforeach ()
    foreach (file ${math_SRC})
        list(APPEND ${libname}_SRC math/${file})
    endforeach ()
    foreach (file ${sky_SRC})
        list(APPEND ${libname}_SRC sky/${file})
    endforeach ()
    foreach (file ${station_SRC})
        list(APPEND ${libname}_SRC station/${file})
    endforeach ()
    foreach (file ${utility_SRC})
        list(APPEND ${libname}_SRC utility/${file})
    endforeach ()

    set(CUDA_GENERATED_OUTPUT_DIR
        ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${libname}.dir/src)

    cuda_add_library(${libname} ${${libname}_SRC})
    target_link_libraries(${libname} ${CUDA_curand_LIBRARY})

    if (OSKAR_USE_CBLAS)
        target_link_libraries(${libname} ${OSKAR_CBLAS})
    endif ()
    if (OSKAR_USE_LAPACK)
        target_link_libraries(${libname} ${OSKAR_LAPACK})
    endif ()
    if (MSVC)
        set_target_properties(${libname} PROPERTIES
            COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
            SOVERSION ${OSKAR_VERSION})
    else ()
        if (APPLE)
            set_target_properties(${libname} PROPERTIES
                INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
                INSTALL_RPATH_USE_LINK_PATH TRUE
                COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
                LINK_FLAGS "${OpenMP_CXX_FLAGS}"
                SOVERSION ${OSKAR_VERSION} 
                VERSION ${OSKAR_VERSION})
        else()
            set_target_properties(${libname} PROPERTIES
                INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
                INSTALL_RPATH_USE_LINK_PATH TRUE
                COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
                LINK_FLAGS "${OpenMP_CXX_FLAGS}"
                SOVERSION ${OSKAR_VERSION} 
                VERSION ${OSKAR_VERSION})
                endif()
    endif()
    
#    set(props
#        LINK_LIBRARIES
#    )
#    foreach (p ${props})
#        get_target_property(TEMP ${libname} ${p})
#        message(">>>>>> ${p} = ${TEMP}")
#    endforeach()

    install(TARGETS ${libname} DESTINATION ${OSKAR_LIB_INSTALL_DIR})

    cuda_build_clean_target()
endif (CUDA_FOUND)

# === Build other oskar libraries.
add_subdirectory(apps)
add_subdirectory(fits)
add_subdirectory(matlab)
add_subdirectory(ms)
#add_subdirectory(python)
add_subdirectory(widgets)

# === Build external libraries (gtest)
add_subdirectory(extern)

# === Install header tree.
if (NOT ${CMAKE_INSTALL_PREFIX} MATCHES "/usr/local")
    string(REGEX MATCH "[^/|\\][a-zA-z0-9|_|-]+$" bin_dir ${CMAKE_BINARY_DIR})
    string(REGEX MATCH "[^/|\\]([a-zA-z0-9|_|-]+$)" install_dir ${CMAKE_INSTALL_PREFIX})
    string(COMPARE EQUAL ${CMAKE_INSTALL_PREFIX} ${CMAKE_BINARY_DIR} install_in_build_dir)
    if ($install_in_build_dir)
        set(install_include_dir "${bin_dir}/include")
    else ()
        set(install_include_dir "${bin_dir}/${install_dir}/include")
    endif()
else()
    set(install_include_dir "${CMAKE_INSTALL_PREFIX}/include")
endif ()
install(DIRECTORY "${OSKAR_SOURCE_DIR}/."
    DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
    FILES_MATCHING PATTERN "oskar_*.h"
    PATTERN ${install_include_dir} EXCLUDE
    PATTERN build EXCLUDE
    PATTERN release EXCLUDE
    PATTERN debug EXCLUDE
    PATTERN .settings EXCLUDE
    PATTERN cmake EXCLUDE
    PATTERN matlab EXCLUDE
    PATTERN dierckx EXCLUDE
    PATTERN src  EXCLUDE
    PATTERN test EXCLUDE
    PATTERN extern EXCLUDE
    PATTERN .svn EXCLUDE)

# === Include packaging script.
include(cmake/oskar_packaging.cmake)

