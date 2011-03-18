#
# /src (top level CMakeLists.txt)
#

# === Set the project name and specify the required CMAKE version.
project(oskar-lib CXX C)
cmake_minimum_required(VERSION 2.6.4 FATAL_ERROR)

# === Project version.
# (todo)

# === Append the src/cmake directory to the module path.
list(INSERT CMAKE_MODULE_PATH 0 ${oskar-lib_SOURCE_DIR}/cmake)

# === Install directories for oskar-lib components.
# Note: the install directory prefix is set by CMAKE_INSTALL_PRFEIX.
if (NOT OSKAR_LIB_INSTALL_DIR)
    set(OSKAR_LIB_INSTALL_DIR "lib")
endif()

if (NOT OSKAR_BIN_INSTALL_DIR)
    set(OSKAR_BIN_INSTALL_DIR "bin")
endif()

# === Set compiler options.
include(compilerOptions)

# === Set dependencies.
include(dependencies)

# === Recurse into sub-diretories.
#add_subdirectory(math)
#add_subdirectory(data)
#add_subdirectory(utility)
#add_subdirectory(mpi)
#add_subdirectory(widgets)
#add_subdirectory(apps)
if (CUDA_FOUND)
    add_subdirectory(cuda)
endif()
if(CASACORE_FOUND)
    #add_subdirectory(ms)
endif()


set(CPACK_GENERATOR "DEB" "TGZ")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "oskar-team")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libcuda, libcudart")
include(CPack)
