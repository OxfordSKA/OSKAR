#
# src/CMakeLists.txt (top level CMakeLists.txt)
#

# ==== Set the project name and specify the required CMAKE version.
project(oskar-lib CXX C Fortran)
cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

# ==== Project version.
set(oskar-lib_VERSION_MAJOR 0)
set(oskar-lib_VERSION_MINOR 0)
set(oskar-lib_VERSION_PATCH 0)
set(oskar-lib_SOVERSION "${oskar-lib_VERSION_MAJOR}.${oskar-lib_VERSION_MINOR}.${oskar-lib_VERSION_PATCH}")

# ==== Append the src/cmake directory to the module path.
list(INSERT CMAKE_MODULE_PATH 0 ${oskar-lib_SOURCE_DIR}/cmake)

# ==== Install directories for oskar-lib components.
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

# ==== Build main oskar library.
if (BUILD_OSKAR)
    add_subdirectory(cuda) # Deprecated.
    add_subdirectory(imaging)
    add_subdirectory(interferometry)
    add_subdirectory(math)
    add_subdirectory(sky)
    add_subdirectory(station)
    add_subdirectory(utility)
    foreach (file ${cuda_SRC}) # Deprecated.
        list(APPEND oskar_SRC cuda/${file})
    endforeach ()
    foreach (file ${imaging_SRC})
        list(APPEND oskar_SRC imaging/${file})
    endforeach ()
    foreach (file ${interferometry_SRC})
        list(APPEND oskar_SRC interferometry/${file})
    endforeach ()
    foreach (file ${math_SRC})
        list(APPEND oskar_SRC math/${file})
    endforeach ()
    foreach (file ${sky_SRC})
        list(APPEND oskar_SRC sky/${file})
    endforeach ()
    foreach (file ${station_SRC})
        list(APPEND oskar_SRC station/${file})
    endforeach ()
    foreach (file ${utility_SRC})
        list(APPEND oskar_SRC utility/${file})
    endforeach ()

    include_directories(${CUDA_INCLUDE_DIRS})
    set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/oskar.dir/src)
    cuda_add_library(oskar ${oskar_SRC})
    target_link_libraries(oskar dierckx ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES} ${CUDA_curand_LIBRARY})
    if (CBLAS_FOUND)
        target_link_libraries(oskar ${CBLAS_LIBRARIES})
    endif ()
    if (LAPACK_FOUND)
        target_link_libraries(oskar ${LAPACK_LIBRARIES})
    endif ()            
    set_target_properties(oskar PROPERTIES
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        LINK_FLAGS "${OpenMP_CXX_FLAGS}"
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
        INSTALL_RPATH_USE_LINK_PATH TRUE
        SOVERSION ${oskar-lib_SOVERSION})
    install(TARGETS oskar DESTINATION ${OSKAR_LIB_INSTALL_DIR})
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
install(DIRECTORY "${oskar-lib_SOURCE_DIR}/."
    DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.h"
    PATTERN "src" EXCLUDE
    PATTERN "test" EXCLUDE
    PATTERN "matlab" EXCLUDE
    PATTERN "python" EXCLUDE
    PATTERN ".svn" EXCLUDE)
































####
#### FIXME: Move all the packaging stuff to a separate cmake script.
####
# include(packaging)
# - move the top level cmake so that we can install scripts, external libs etc.
# - will also need to sort out the install of dependencies.
#    - on windows just provide copies of the dll's with the install...
#    - on linux same? or rely on packages.... (bad idea if we need certain
#      versions of external libs)
################################################################################
# === Packaging (make package) info: dpkg -I oskar-lib-0.0.0-Linux.deb
# http://wiki.clug.org.za/wiki/How_do_I_install_a_.deb_file_I_downloaded_without_compromising_dependencies%3F
# dpkg-scanpackages . /dev/null | gzip -c -9 > Packages.gz
# deb file:///home/debs /
set(CPACK_PACKAGE_VENDOR "Oxford e-Research Centre, University of Oxford.")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "OSKAR developer team.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The Open SKA Radio Telescope library.")
set(CPACK_PACKAGE_VERSION ${oskar-lib_SOVERSION})
if (WIN32)
    # http://www.itk.org/Wiki/CMake:Component_Install_With_CPack
    set(CPACK_GENERATOR "NSIS")
elseif ()
    set(CPACK_GENERATOR "DEB" "TGZ")
    set(CPACK_DEBIAN_PACKAGE_VERSION ${oskar-lib_SOVERSION})
    set(CPACK_DEBIAN_PACKAGE_SECTION "Science")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
    set(CPACK_DEBIAN_PACKAGE_DEPENDS "libcfitsio3 (>=3.0), libqt4-dev (>=4.5)")
endif ()
include(CPack)
