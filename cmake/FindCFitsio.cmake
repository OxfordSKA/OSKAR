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
    PATHS
    /usr/include/cfitsio/
    /usr/include/
)

set(CFITSIO_NAMES cfitsio)

foreach (lib ${CFITSIO_NAMES})
    find_library(CFITSIO_LIBRARY_${lib} NAMES ${lib})
    list(APPEND CFITSIO_LIBRARIES ${CFITSIO_LIBRARY_${lib}})
endforeach (lib ${CFITSIO_NAMES})

# handle the QUIETLY and REQUIRED arguments and set CFITSIO_FOUND to TRUE if.
# all listed variables are TRUE
include(FindPackageHandleCompat)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CFitsio DEFAULT_MSG
    CFITSIO_LIBRARIES CFITSIO_INCLUDE_DIR)

if (NOT CFITSIO_FOUND)
    set(CFITSIO_LIBRARIES)
endif (NOT CFITSIO_FOUND)

mark_as_advanced(CFITSIO_LIBRARIES CFITSIO_INCLUDE_DIR)
