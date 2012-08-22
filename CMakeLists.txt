#
# src/CMakeLists.txt (top level CMakeLists.txt)
#

# === Set the project name and specify the required CMAKE version.
project(OSKAR CXX C)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# === Project version.
set(OSKAR_VERSION_MAJOR 2)
set(OSKAR_VERSION_MINOR 0)
set(OSKAR_VERSION_PATCH 4)
set(OSKAR_VERSION "${OSKAR_VERSION_MAJOR}.${OSKAR_VERSION_MINOR}.${OSKAR_VERSION_PATCH}")

# === Append the src/cmake directory to the module path.
list(INSERT CMAKE_MODULE_PATH 0 ${OSKAR_SOURCE_DIR}/cmake)

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
    set(OSKAR_MATLAB_INSTALL_DIR "share/matlab/toolbox/oskar")
endif()

# === Set dependencies and compiler options.
include(oskar_dependencies)
include(oskar_compiler_options)

# === Include build macros used for apps and mex functions.
include(oskar_build_macros)

# Configure MSVC runtime. NOTE: should this not be in the compiler_options.cmake?
OSKAR_SET_MSVC_RUNTIME()

enable_testing()

# === Set the top level OSKAR source directory in the include path.
include_directories(${OSKAR_SOURCE_DIR})

# === Build main OSKAR library.
if (CUDA_FOUND)
    include_directories(${OSKAR_SOURCE_DIR})
    #set(CUDA_VERBOSE_BUILD ON)

    add_subdirectory(imaging)
    add_subdirectory(interferometry)
    add_subdirectory(math)
    add_subdirectory(sky)
    add_subdirectory(station)
    add_subdirectory(utility)

    set(libname oskar)

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

    set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${libname}.dir/src)
    cuda_add_library(${libname} ${${libname}_SRC})
    target_link_libraries(${libname}
        ${CUDA_LIBRARIES}
        ${CUDA_curand_LIBRARY})
    if (OSKAR_USE_CBLAS)
        target_link_libraries(${libname} ${OSKAR_CBLAS})
    endif ()
    if (OSKAR_USE_LAPACK)
        target_link_libraries(${libname} ${OSKAR_LAPACK})
    endif ()
    if (MSVC)
        set_target_properties(${libname} PROPERTIES
            COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
            INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
            INSTALL_RPATH_USE_LINK_PATH TRUE
            SOVERSION ${OSKAR_VERSION}
        )
    else ()
        set_target_properties(${libname} PROPERTIES
            COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
            LINK_FLAGS "${OpenMP_CXX_FLAGS}"
            INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
            INSTALL_RPATH_USE_LINK_PATH TRUE
            SOVERSION ${OSKAR_VERSION}
        )
    endif()
    install(TARGETS ${libname} DESTINATION ${OSKAR_LIB_INSTALL_DIR})
    cuda_build_clean_target()
endif (CUDA_FOUND)

# === Build other oskar libraries.
add_subdirectory(apps)
add_subdirectory(fits)
add_subdirectory(matlab)
add_subdirectory(ms)
add_subdirectory(widgets)

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
    #message("== Excluding install include directory '${install_include_dir}'")
else()
    set(install_include_dir "${CMAKE_INSTALL_PREFIX}/include")
endif ()
install(DIRECTORY "${OSKAR_SOURCE_DIR}/."
    DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.h"
    PATTERN ${install_include_dir} EXCLUDE
    PATTERN build EXCLUDE
    PATTERN .settings EXCLUDE
    PATTERN cmake EXCLUDE
    PATTERN matlab EXCLUDE
    PATTERN dierckx EXCLUDE
    PATTERN src  EXCLUDE
    PATTERN test EXCLUDE
    PATTERN .svn EXCLUDE)

# === Include packaging script.
include(oskar_packaging)

