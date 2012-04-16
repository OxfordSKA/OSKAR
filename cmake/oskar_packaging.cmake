#
# OSKAR CMAKE packaging script
#
# Description:
# Adds tagets: 
#   1. package (binary package)
#   2. package_source (source package) 
# 
# Notes:
#
# http://www.cmake.org/Wiki/CMake:CPackConfiguration
#
# http://www.cmake.org/Wiki/CMakeUserUseDebian
# 
# info: dpkg -I oskar-lib-0.0.0-Linux.deb
#
# http://wiki.clug.org.za/wiki/How_do_I_install_a_.deb_file_I_downloaded_without_compromising_dependencies%3F
# dpkg-scanpackages . /dev/null | gzip -c -9 > Packages.gz
# deb file:///home/debs /
#

SET(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/readme.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OSKAR-${OSKAR_VERSION}")
set(CPACK_PACKAGE_VENDOR "University of Oxford, Oxford e-Research Centre.")
SET(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/copying.txt")
SET(CPACK_PACKAGE_VERSION_MAJOR "${OSKAR_VERSION_MAJOR}")
SET(CPACK_PACKAGE_VERSION_MINOR "${OSKAR_VERSION_MINOR}")
SET(CPACK_PACKAGE_VERSION_PATCH "${OSKAR_VERSION_PATCH}")
set(CPACK_PACKAGE_VERSION "${OSKAR_VERSION}-beta1")

set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)

if (WIN32)
    # http://www.itk.org/Wiki/CMake:Component_Install_With_CPack
    set(CPACK_GENERATOR "NSIS")
elseif (UNIX)
    set(CPACK_GENERATOR "DEB;TGZ")
    set(CPACK_SOURCE_GENERATOR "TGZ")
    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "OSKAR developer team.")
    set(CPACK_DEBIAN_PACKAGE_VERSION ${OSKAR_VERSION})
    set(CPACK_DEBIAN_PACKAGE_SECTION "Science")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
    #set(CPACK_DEBIAN_PACKAGE_DEPENDS "libcfitsio3 (>=3.0), libqt4-dev (>=4.5)")
endif ()
include(CPack)
