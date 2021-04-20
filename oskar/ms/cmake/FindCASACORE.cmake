# - Find casacore
#==============================================================================
# Find the native CASACORE includes and library
#
#  CASACORE_INCLUDE_DIR  - where to find casacore.h, etc.
#  CASACORE_LIBRARY_PATH - Specify to choose a non-standard location to
#                          search for libraries
#  CASACORE_LIBRARIES    - List of casacore libraries.
#  CASACORE_FOUND        - True if casacore found.
#==============================================================================
#
# Casacore dependencies between sub-packages:
# http://usg.lofar.org/wiki/doku.php?id=software:packages:casacore:dependency_of_the_packages
#
#==============================================================================
#

if (1)
# Enable only modules that are required for libcasa_tables.
# Avoids pulling in FORTRAN runtime, LAPACK, and unneeded casacore modules.
set(casacore_modules
    casa_tables
    casa_casa
)

if (NOT WIN32)
    find_path(CASACORE_INCLUDE_DIR Tables.h
        HINTS ${CASACORE_INC_DIR}
        PATH_SUFFIXES casacore/tables tables)
    if (CASACORE_INCLUDE_DIR)
        get_filename_component(CASACORE_INCLUDE_DIR ${CASACORE_INCLUDE_DIR} DIRECTORY)
        get_filename_component(CASACORE_INCLUDE_DIR ${CASACORE_INCLUDE_DIR} DIRECTORY)
        foreach (module ${casacore_modules})
            find_library(CASACORE_LIBRARY_${module} NAMES ${module}
                HINTS ${CASACORE_LIB_DIR}
                PATHS ENV CASACORE_LIBRARY_PATH
                PATH_SUFFIXES lib)
            mark_as_advanced(CASACORE_LIBRARY_${module})
            list(APPEND CASACORE_LIBRARIES ${CASACORE_LIBRARY_${module}})
        endforeach()
    endif()
endif()

else()
# Old version.
# Enable only modules that are required for libcasa_ms.
set(casacore_modules
    casa_ms
    casa_measures
    casa_tables
    casa_scimath
    casa_scimath_f
    casa_casa
)

# mmm this only works by luck by the looks of things...!
find_package(LAPACK QUIET)
if (LAPACK_FOUND)
    set(CASACORE_LINKER_FLAGS ${LAPACK_LINKER_FLAGS})
    find_path(CASACORE_INCLUDE_DIR MeasurementSets.h
        HINTS ${CASACORE_INC_DIR}
        PATH_SUFFIXES casacore/ms ms)
    if (CASACORE_INCLUDE_DIR)
        get_filename_component(CASACORE_INCLUDE_DIR ${CASACORE_INCLUDE_DIR} DIRECTORY)
        get_filename_component(CASACORE_INCLUDE_DIR ${CASACORE_INCLUDE_DIR} DIRECTORY)
        foreach (module ${casacore_modules})
            find_library(CASACORE_LIBRARY_${module} NAMES ${module}
                HINTS ${CASACORE_LIB_DIR}
                PATHS ENV CASACORE_LIBRARY_PATH
                PATH_SUFFIXES lib)
            mark_as_advanced(CASACORE_LIBRARY_${module})
            list(APPEND CASACORE_LIBRARIES ${CASACORE_LIBRARY_${module}})
        endforeach()
        list(APPEND CASACORE_LIBRARIES ${LAPACK_LIBRARIES})
    endif()
endif()
endif(1)

# handle the QUIETLY and REQUIRED arguments and set CASACORE_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CASACORE DEFAULT_MSG
    CASACORE_LIBRARIES CASACORE_INCLUDE_DIR)

if (NOT CASACORE_FOUND)
    set(CASACORE_LIBRARIES)
endif()
