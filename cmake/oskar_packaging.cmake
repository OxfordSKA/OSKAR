#
# OSKAR CMAKE packaging script
#
# Description:
# Adds tagets: 
#   1. dist (build the source package with included version file)
# 
#   2. package (binary package)
#   3. package_source (source package) 
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

# Find the Subversion revision.
include(oskar_build_macros)
get_svn_revision(${OSKAR_SOURCE_DIR} svn_revision)

set(CPACK_PACKAGE_DESCRIPTION_FILE "${CMAKE_CURRENT_SOURCE_DIR}/readme.txt")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "OSKAR-${OSKAR_VERSION}")
set(CPACK_PACKAGE_VENDOR "University of Oxford, Oxford e-Research Centre.")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/copying.txt")
set(CPACK_PACKAGE_VERSION_MAJOR "${OSKAR_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${OSKAR_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${OSKAR_VERSION_PATCH}")
if (svn_revision)
    set(CPACK_PACKAGE_VERSION "${OSKAR_VERSION}-beta-r${svn_revision}")
else()
    set(CPACK_PACKAGE_VERSION "${OSKAR_VERSION}-beta")
endif()
set(CPACK_INCLUDE_TOPLEVEL_DIRECTORY ON)

if (WIN32)
    # http://www.itk.org/Wiki/CMake:Component_Install_With_CPack
    set(CPACK_GENERATOR "NSIS")
elseif (UNIX)
    #set(CPACK_GENERATOR "DEB;TGZ")
    set(CPACK_GENERATOR "TGZ")
    #if (APPLE )
    #    set(CPACK_GENERATOR "Bundle")
    #    set(CPACK_BUNDLE_NAME "oskar")
    #    set(CPACK_BUNDLE_PLIST
    #endif()
    set(CPACK_SOURCE_GENERATOR "ZIP")

    set(CPACK_DEBIAN_PACKAGE_MAINTAINER "OSKAR developer team.")
    set(CPACK_DEBIAN_PACKAGE_VERSION "${OSKAR_VERSION}-beta")
    set(CPACK_DEBIAN_PACKAGE_SECTION "Science")
    set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
    #set(CPACK_DEBIAN_PACKAGE_DEPENDS "libcfitsio3 (>=3.0), libqt4-dev (>=4.5) ")
endif ()
include(CPack)

add_custom_target(write_version_file
    COMMAND ${CMAKE_COMMAND}
    ARGS 
    -DOSKAR_SOURCE_DIR=${OSKAR_SOURCE_DIR}
    -DVERSION=${CPACK_PACKAGE_VERSION}
    -P ${OSKAR_SOURCE_DIR}/cmake/oskar_write_version_file.cmake
    COMMENT "Writing version file"
    VERBATIM)

add_custom_target(dist 
    COMMAND ${CMAKE_MAKE_PROGRAM} package_source
    COMMENT "Packaging Source files"
    VERBATIM)
    
add_dependencies(dist write_version_file)

