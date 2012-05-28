# - Find OSKAR2
# =============================================================================
#
# This module can be used to find OSKAR2, and defines a number of key
# variables required for building and linking a project using this library.
#
# Typical usage could be something like:
#
#   find_package(OSKAR2 2.0.0 COMPONENTS oskar_apps oskar_ms oskar_widgets)
#   include_directories(${OSKAR_INCLUDES})
#   add_executable(myexe main.cpp)
#   target_link_libraries(myexe ${OSKAR_LIBRARIES})
#
# =============================================================================
# The following is a list of variables set:
#
#  OSKAR_FOUND                  True if OSKAR found.
#
#  OSKAR_VERSION_MAJOR          The major version of OSKAR found.
#  OSKAR_VERSION_MINOR          The minor version of OSKAR found.
#  OSKAR_VERSION_PATCH          The patch version of OSKAR found.
#
#  OSKAR_MS_FOUND               True if oskar_ms module was found.
#  OSKAR_WIDGETS_FOUND          True if oskar_widgets module was found.
#
#  OSKAR_INCLUDES               List of paths to all include directories.
#  OSKAR_LIBRARY_DIR            Path to "lib" of OSKAR.
#
#  OSKAR_LIBRARIES              Complete list of libararies for using OSKAR.
#
#  OSKAR_MS_LIBRARY             The oskar_ms library.
#  OSKAR_WIDGETS_LIBRARY        The oskar_widgets library.
#
# FIXME: How to deal with external libs?
#=======================================
# -- link tables should be set up correctly for shared version
# -- what about static version?
# -- should still check for libs to get correct error messages before linking?


##############################
#
# General variables.
#
##############################
set(OSKAR_MODULES
    apps
    ms
    fits
    widgets
)

##############################
#
# Version variables.
#
##############################

# Check version against that specified.
#set(OSKAR_INSTALLED_VERSION_TOO_OLD false)

#message("OSKAR MIN VERSION ${OSKAR_MIN_VERSION}")

# We need at least version 0.0.0!
#if (NOT OSKAR_MIN_VERSION)
#    set(OSKAR_MIN_VERSION "0.0.0")
#endif ()

#message("OSKAR MIN VERSION ${OSKAR_MIN_VERSION}")

# Now parse the parts of the user given version string into variables
#string(REGEX MATCH "^[0-9]+\\.[0-9]+\\.[0-9]+" req_oskar_major_vers
#    "${OSKAR_MIN_VERSION}")
#if (NOT req_oskar_major_ver)
#    message(FATAL_ERROR "Invalid version of OSKAR string was given:
#            \"${OSKAR_MIN_VERSION}\", expected e.g. \"0.1.2\"")
#endif ()

# Now parse the parts of the user given version string into variables
#string(REGEX REPLACE "^([0-9]+)\\.[0-9]+\\.[0-9]+" "\\1" req_oskar_major_vers
#     "${OSKAR_MIN_VERSION}")
#string(REGEX REPLACE "^[0-9]+\\.([0-9])+\\.[0-9]+" "\\1" req_oskar_minor_vers
#    "${OSKAR_MIN_VERSION}")
#string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+)" "\\1" req_oskar_patch_vers
#    "${OSKAR_MIN_VERSION}")

# Get the version string from the library...?


##############################
#
# Include variables.
#
##############################

find_path(OSKAR_INCLUDE_DIR oskar.h
          PATHS
          /usr/include/oskar
          /usr/local/include/oskar
          /opt/include/oskar
)

set(OSKAR_INCLUDES ${OSKAR_INCLUDE_DIR})

mark_as_advanced(OSKAR_INCLUDE_DIR)




##############################
#
# Library variables.
#
##############################

find_library(OSKAR_LIBRARY NAMES oskar
    PATHS
    /usr/lib/
    /usr/local/lib/
    /usr/local/oskar/lib/
    /usr/oskar/lib/
    )
if (${OSKAR_LIBRARY} MATCHES OSKAR_LIBRARY-NOTFOUND)
    #message("oops")
else()
    list(APPEND OSKAR_LIBRARIES ${OSKAR_LIBRARY})
endif ()

foreach (module ${OSKAR_MODULES})
    string(TOUPPER ${module} _upper_oskar_module)
    set(lib_name "oskar_${module}")
    find_library(OSKAR_${_upper_oskar_module}_LIBRARY NAMES ${lib_name}
        PATHS
        /usr/lib/
        /usr/local/lib/
        /usr/local/oskar/lib/
        /usr/oskar/lib/
        )
    #message("----- ${lib_name}: ${OSKAR_${_upper_oskar_module}_LIBRARY}")
    if (${OSKAR_${_upper_oskar_module}_LIBRARY} MATCHES OSKAR_${_upper_oskar_module}_LIBRARY-NOTFOUND)
        #message("oops")
    else()
        list(APPEND OSKAR_LIBRARIES ${OSKAR_${_upper_oskar_module}_LIBRARY})
    endif ()
endforeach ()

#message("**** ${OSKAR_LIBRARIES}")

mark_as_advanced(OSKAR_LIBRARIES)



##############################
#
# Check external dependencies.
#
##############################

# TODO!




##############################
#
# Decide if OSKAR was found.
#
##############################

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OSKAR DEFAULT_MSG
    OSKAR_LIBRARIES OSKAR_INCLUDE_DIR)


