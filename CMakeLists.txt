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
    set(OSKAR_LIB_INSTALL_DIR "${CMAKE_PROJECT_NAME}/lib")
endif()

if (NOT OSKAR_INCLUDE_INSTALL_DIR)
    set(OSKAR_INCLUDE_INSTALL_DIR "${CMAKE_PROJECT_NAME}/include")
endif()

if (NOT OSKAR_BIN_INSTALL_DIR)
    set(OSKAR_BIN_INSTALL_DIR "${CMAKE_PROJECT_NAME}/bin")
endif()

# === Set dependencies.
include(dependencies)

# === Set compiler options.
include(compilerOptions)

# === Recurse into sub-diretories.
#add_subdirectory(apps)
#add_subdirectory(data)
add_subdirectory(cuda)
if (NOT WIN32) # Note: need to fix the __declspec(dllexport) for windows dll's
    add_subdirectory(math)
    add_subdirectory(ms)
    add_subdirectory(utility)
    add_subdirectory(widgets)
endif ()


# === Packaging (make package) info: dpkg -I oskar-lib-0.0.0-Linux.deb
# http://wiki.clug.org.za/wiki/How_do_I_install_a_.deb_file_I_downloaded_without_compromising_dependencies%3F
# dpkg-scanpackages . /dev/null | gzip -c -9 > Packages.gz
# deb file:///home/debs /
set(CPACK_GENERATOR "DEB" "TGZ")
set(CPACK_PACKAGE_VENDOR "Oxford e-Research Centre, University of Oxford.")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER "OSKAR developer team.")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The Open SKA Radio Telescope library.")
set(CPACK_PACKAGE_VERSION ${oskar-lib_SOVERSION})
set(CPACK_DEBIAN_PACKAGE_VERSION ${oskar-lib_SOVERSION})
set(CPACK_DEBIAN_PACKAGE_SECTION "Science")
set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
set(CPACK_DEBIAN_PACKAGE_DEPENDS "libcfitsio3 (>=3.0), libqt4-dev (>=4.5)")
include(CPack)
