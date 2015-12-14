# Find cfitsio.
# Find the native CFITSIO includes and library
#
# Defines the following variables:
#
#  CFITSIO_FOUND         = True if cfitsio found
#  CFITSIO_LIBRARIES     = Set of libraries required for linking
#  CFITSIO_INCLUDE_DIR   = Directory where to find fitsio.h
#

find_path(CFITSIO_INCLUDE_DIR fitsio.h
    HINTS ${CFITSIO_INC_DIR}
    PATHS
    /usr/include/cfitsio
    /apps/libs/cfitsio/gnu/3310/include
    "C:\\cfitsio")

set(CFITSIO_NAMES cfitsio)

foreach (lib ${CFITSIO_NAMES})
    find_library(CFITSIO_LIBRARY_${lib} NAMES ${lib}
        HINTS ${CFITSIO_LIB_DIR} ${CFITSIO_INCLUDE_DIR}/../lib
        PATHS /apps/libs/cfitsio/gnu/3310/lib)
    list(APPEND CFITSIO_LIBRARIES ${CFITSIO_LIBRARY_${lib}})
endforeach (lib ${CFITSIO_NAMES})

# handle the QUIETLY and REQUIRED arguments and set CFITSIO_FOUND to TRUE if
# all listed variables are TRUE
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CFitsio DEFAULT_MSG
    CFITSIO_LIBRARIES CFITSIO_INCLUDE_DIR)

if (NOT CFITSIO_FOUND)
    set(CFITSIO_LIBRARIES)
endif (NOT CFITSIO_FOUND)

mark_as_advanced(CFITSIO_LIBRARIES CFITSIO_INCLUDE_DIR)
