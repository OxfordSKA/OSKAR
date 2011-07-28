#
# src/CMakeLists.txt (top level CMakeLists.txt)
#
#==============================================================================

# === Set the project name and specify the required CMAKE version.
project(oskar-lib CXX C)
cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

# === Project version.
set(oskar-lib_VERSION_MAJOR 0)
set(oskar-lib_VERSION_MINOR 0)
set(oskar-lib_VERSION_PATCH 0)
set(oskar-lib_SOVERSION "${oskar-lib_VERSION_MAJOR}.${oskar-lib_VERSION_MINOR}.${oskar-lib_VERSION_PATCH}")

# === Append the src/cmake directory to the module path.
list(INSERT CMAKE_MODULE_PATH 0 ${oskar-lib_SOURCE_DIR}/cmake)



# === Install directories for oskar-lib components.
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
    set(OSKAR_MATLAB_INSTALL_DIR "share/oskar/matlab")
endif()

# === Set dependencies.
include(dependencies)

# === Set compiler options.
include(compilerOptions)

# === Recurse into sub-diretories.
add_subdirectory(cuda)
if (NOT WIN32) # Note: need to fix the __declspec(dllexport) for windows dll's
    add_subdirectory(beamforming)
    add_subdirectory(sky)
    add_subdirectory(imaging)
    add_subdirectory(math)
    add_subdirectory(modules)
    add_subdirectory(ms)
    add_subdirectory(utility)
    add_subdirectory(widgets)
    add_subdirectory(interferometry)
endif ()




# ==== Build single oskar library.
foreach (file ${beamforming_SRC})
    list(APPEND oskar_SRC beamforming/${file})
endforeach ()
foreach (file ${cuda_SRC})
    list(APPEND oskar_SRC cuda/${file})
endforeach ()
foreach (file ${interferometry_SRC})
    list(APPEND oskar_SRC interferometry/${file})
endforeach ()
foreach (file ${math_SRC})
    list(APPEND oskar_SRC math/${file})
endforeach ()
foreach (file ${modules_SRC})
    list(APPEND oskar_SRC modules/${file})
endforeach ()
foreach (file ${sky_SRC})
    list(APPEND oskar_SRC sky/${file})
endforeach ()
foreach (file ${utility_SRC})
    list(APPEND oskar_SRC utility/${file})
endforeach ()

message("=============")
foreach (file ${oskar_SRC})
    message(">>>>>>>>>> ${file}")
endforeach()
message("=============")

include_directories(${oskar-lib_SOURCE_DIR}/beamforming)
include_directories(${oskar-lib_SOURCE_DIR}/cuda)
include_directories(${oskar-lib_SOURCE_DIR}/interferometry)
include_directories(${oskar-lib_SOURCE_DIR}/math)
include_directories(${oskar-lib_SOURCE_DIR}/modules)
include_directories(${oskar-lib_SOURCE_DIR}/sky)
include_directories(${oskar-lib_SOURCE_DIR}/utility)


set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/oskar.dir/src)
cuda_add_library(oskar ${oskar_SRC})
set_target_properties(oskar PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
target_link_libraries(oskar
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_LIBRARIES})
set_target_properties(oskar PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
    INSTALL_RPATH_USE_LINK_PATH TRUE)

install(TARGETS oskar LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR})
set_target_properties(oskar PROPERTIES SOVERSION ${oskar-lib_SOVERSION})
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
