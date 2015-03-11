#
# Sets:
#   OSKAR_VERSION_ID
#   OSKAR_VERSION_MAJOR
#   OSKAR_VERSION_MINOR
#   OSKAR_VERSION_PATCH
#   OSKAR_VERSION_SUFFIX
#   OSKAR_SVN_REVISION
#   OSKAR_OSKAR_VERSION_STR
#

if (NOT OSKAR_VERSION_ID)

    # Modify the following lines for updating the OSKAR version number.
    set(OSKAR_VERSION_ID "0x020600")
    set(OSKAR_VERSION_MAJOR 2)
    set(OSKAR_VERSION_MINOR 6)
    set(OSKAR_VERSION_PATCH 0)
    set(OSKAR_VERSION_SUFFIX "trunk")

    # Try to find the subversion revision.
    find_package(Subversion QUIET)
    if (SUBVERSION_FOUND)
        get_filename_component(SVN_PATH ${PROJECT_SOURCE_DIR} REALPATH)
        Subversion_WC_INFO(${SVN_PATH} OSKAR_SVN)
        if (OSKAR_SVN_WC_REVISION)
            set(OSKAR_SVN_REVISION ${OSKAR_SVN_WC_REVISION})
        endif()
    endif()

    # Construct OSKAR_VERSION_STR.
    set(OSKAR_VERSION "${OSKAR_VERSION_MAJOR}.${OSKAR_VERSION_MINOR}.${OSKAR_VERSION_PATCH}")
    set(OSKAR_VERSION_STR "${OSKAR_VERSION}")
    if (OSKAR_VERSION_SUFFIX)
        set(OSKAR_VERSION_STR "${OSKAR_VERSION}-${OSKAR_VERSION_SUFFIX}")
        if (OSKAR_SVN_REVISION)
            set(OSKAR_VERSION_STR "${OSKAR_VERSION_STR} r${OSKAR_SVN_REVISION}")
        endif()
        if (CMAKE_BUILD_TYPE MATCHES debug)
            set(OSKAR_VERSION_STR "${OSKAR_VERSION_STR} -- debug --")
        endif()
    endif()

endif()
