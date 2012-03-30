#
# src/CMakeLists.txt (top level CMakeLists.txt)
#

# ==== Set the project name and specify the required CMAKE version.
project(OSKAR CXX C Fortran)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# ==== Project version.
set(OSKAR_VERSION_MAJOR 2)
set(OSKAR_VERSION_MINOR 0)
set(OSKAR_VERSION_PATCH 0)
set(OSKAR_VERSION "${OSKAR2_VERSION_MAJOR}.${OSKAR2_VERSION_MINOR}.${OSKAR2_VERSION_PATCH}")

# ==== Append the src/cmake directory to the module path.
list(INSERT CMAKE_MODULE_PATH 0 ${OSKAR_SOURCE_DIR}/cmake)

# ==== Setup install directory paths
# Note: the install directory prefix is set by CMAKE_INSTALL_PRFEIX.
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

# ==== Set the libraries that are built by default.
if (NOT DEFINED BUILD_OSKAR)
    set(BUILD_OSKAR ON)
endif ()

if (NOT DEFINED BUILD_OSKAR_MS)
    set(BUILD_OSKAR_MS ON)
endif ()

if (NOT DEFINED BUILD_OSKAR_IMAGING)
    set(BUILD_OSKAR_IMAGING ON)
endif ()

if (NOT DEFINED BUILD_OSKAR_WIDGETS)
    set(BUILD_OSKAR_WIDGETS ON)
endif ()

if (NOT DEFINED BUILD_OSKAR_APPS)
    set(BUILD_OSKAR_APPS ON)
endif()

if (NOT DEFINED BUILD_OSKAR_FITS)
    set(BUILD_OSKAR_FITS ON)
endif ()

# ==== Set dependencies and compiler options.
include(oskar_dependencies)
include(oskar_compiler_options)

enable_testing()

# Build external libraries.
add_subdirectory(extern)

# ==== Build main OSKAR library.
if (BUILD_OSKAR)
    #add_subdirectory(cuda) # Deprecated.
    add_subdirectory(imaging)
    add_subdirectory(interferometry)
    add_subdirectory(math)
    add_subdirectory(sky)
    add_subdirectory(station)
    add_subdirectory(utility)
    
    set(libname oskar)
    
    #foreach (file ${cuda_SRC}) # Deprecated.
    #    list(APPEND ${libname}_SRC cuda/${file})
    #endforeach ()
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

    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/${libname}.dir/src)
    cuda_add_library(${libname} ${${libname}_SRC})
    target_link_libraries(${libname} dierckx ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY})
    if (OSKAR_USE_CBLAS)
        target_link_libraries(${libname} ${OSKAR_CBLAS})
    endif ()
    if (OSKAR_USE_LAPACK)
        target_link_libraries(${libname} ${OSKAR_LAPACK})
    endif ()
    set_target_properties(${libname} PROPERTIES
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        LINK_FLAGS "${OpenMP_CXX_FLAGS}"
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
        INSTALL_RPATH_USE_LINK_PATH TRUE
        SOVERSION ${OSKAR_VERSION})
    install(TARGETS ${libname} DESTINATION ${OSKAR_LIB_INSTALL_DIR})
    cuda_build_clean_target()
endif (BUILD_OSKAR)


# ==== Build other oskar libraries.
if (BUILD_OSKAR_MS)
    add_subdirectory(ms)
endif ()
if (BUILD_OSKAR_WIDGETS)
    add_subdirectory(widgets)
endif ()
if (BUILD_OSKAR_APPS)
    add_subdirectory(apps)
endif ()
if (BUILD_OSKAR_FITS)
    add_subdirectory(fits)
endif ()


# ==== Install header tree.
install(DIRECTORY "${OSKAR_SOURCE_DIR}/."
    DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.h"
    PATTERN "src" EXCLUDE
    PATTERN "test" EXCLUDE
    PATTERN "matlab" EXCLUDE
    PATTERN "python" EXCLUDE
    PATTERN ".svn" EXCLUDE)

# Include packaging options script.
include(oskar_packaging)
