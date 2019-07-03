# - Find casacore
#==============================================================================
# Find the native CASACORE includes and library
#
#  CASACORE_INCLUDE_DIR  - where to find casacore.h, etc.
#  CASACORE_LIBRARY_PATH - Specify to choose a non-standard location to
#                          search for libraries
#  CASACORE_LIBRARIES    - List of libraries when using casacore.
#  CASACORE_FOUND        - True if casacore found.
#==============================================================================
#
# Required external packages:
#   gfortran
#   cfitsio3
#   wcs
#   blas
#   fftw3
#   fftw3f
#
# Casacore dependencies between sub-packages:
# http://usg.lofar.org/wiki/doku.php?id=software:packages:casacore:dependency_of_the_packages
#
#==============================================================================
#
# Issues:
#   - Near lack of handling of external libraries.
#   - No handling of interdependance of modules.
#   - Exporting of module libraries? (with useful names)
#   - Include dir not set? (module include directories?)
#   - Single library mode?
#   - Library versions?
#
#==============================================================================

# NOTE: This order is important for static version of the library... don't mess with it!
# --> Enabled only modules that required for libcasa_ms
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

# handle the QUIETLY and REQUIRED arguments and set CASACORE_FOUND to TRUE if
# all listed variables are TRUE
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CASACORE DEFAULT_MSG
    CASACORE_LIBRARIES CASACORE_INCLUDE_DIR)

if (NOT CASACORE_FOUND)
    set(CASACORE_LIBRARIES)
endif()
