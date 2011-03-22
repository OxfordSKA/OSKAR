# - Find oskar-lib
# =============================================================================
#
# This module and be used to find oskar-lib and define a number of key variables
# required for building and linking a project using this library.
#
# Typical usage could be something like:
#
#   find_package(OskarLib 0.1 COMPONENTS oskar_widgets oskar_cuda)
#   add_executable(myexe main.cpp)
#   target_link_libraries(myexe ${OSKAR_LIBRARIES})
#
# =============================================================================
# The following is a list of variables set:
#
#  OSKAR_FOUND                  True if oskar-lib found.
#
#  OSKAR_VERSION_MAJOR          The major version of oskar-lib found.
#  OSKAR_VERSION_MINOR          The minor version of oskar-lib found.
#  OSKAR_VERSION_PATCH          The patch version of oskar-lib found.
#
#  OSKAR_CUDA_FOUND             True if oskar_cuda module was found.
#  OSKAR_MS_FOUND               True if oskar_ms module was found.
#  OSKAR_WIDGETS_FOUND          True if oskar_widgets module was found.
#  OSKAR_MATH_FOUND             True if oskar_math module was found.
#
#  OSKAR_INCLUDES               List of paths to all include directories.
#
#  OSKAR_INCLUDE_DIR            Path to "include" of oskar-lib.
#  OSKAR_CUDA_INCLUDE_DIR       Path to "include/cuda".
#  OSKAR_MS_INCLUDE_DIR         Path to "include/ms".
#  OSKAR_WIDGETS_INCLUDE_DIR    Path to "include/widgets".
#  OSKAR_MATH_INCLUDE_DIR       Path to "include/math".
#
#  OSKAR_LIBRARY_DIR            Path to "lib" of oskar-lib.
#
#  OSKAR_LIBRARIES              Complete list of libararies for using oskar-lib.
#
#  OSKAR_CUDA_LIBRARY           The oskar_cuda library.
#  OSKAR_MS_LIBRARY             The oskar_ms library.
#  OSKAR_MATH_LIBRARY           The oskar_math library.
#  OSKAR_WIDGETS_LIBRARY        The oskar_widgets library.
#
# FIXME: mmm how to deal with external libs...
#=========================================
# -- link tables should be set up correctly for sharded version
# -- what about static version...
# -- should still check for libs to get correct error messages before linking?


##############################
#
# General variables.
#
##############################
set(OSKAR_MODULES
    ms
    cuda
    math
    widgets
    widgets_plotting
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
#    message(FATAL_ERROR "Invalid version of oskar-lib string was given:
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

find_path(OSKAR_INCLUDE_DIR oskar
          PATHS
          /usr/include/
          /usr/local/include/
)



foreach (module ${OSKAR_MODULES})
    string(TOUPPER ${module} _upper_oskar_module)
    find_path(OSKAR_${_upper_oskar_module}_INCLUDE_DIR ${module}
        PATHS
        ${OSKAR_INCLUDE_DIR}
    )
endforeach ()

mark_as_advanced(OSKAR_INCLUDE_DIR)




##############################
#
# Library variables.
#
##############################

foreach (module ${OSKAR_MODULES})
    string(TOUPPER ${module} _upper_oskar_module)
    set(lib_name "oskar_${module}")
    find_library(OSKAR_${_upper_oskar_module}_LIBRARY NAMES ${lib_name})
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
# Decide if oskar-lib was found.
#
##############################

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OSKAR DEFAULT_MSG
    OSKAR_LIBRARIES OSKAR_INCLUDE_DIR)


