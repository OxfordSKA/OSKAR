# - Find CBLAS
#
# This module defines
#  CBLAS_LIBRARIES
#  CBLAS_FOUND
#
# Also defined, but not for general use are
#  CBLAS_LIBRARY, where to find the library.


#set(BLAS_FIND_REQUIRED true)

set(CBLAS_NAMES 
    cblas 
    gslcblas
)

find_library(CBLAS_LIBRARY
    NAMES ${CBLAS_NAMES}
    HINTS ${CBLAS_LIB_DIR}
    PATHS
    /usr/lib64/atlas
    /usr/lib/atlas
    /usr/local/lib64/atlas
    /usr/local/lib/atlas
    /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
)

if (CBLAS_LIBRARY)
    set(CBLAS_LIBRARIES ${CBLAS_LIBRARY})
endif (CBLAS_LIBRARY)

# handle the QUIETLY and REQUIRED arguments and set CBLAS_FOUND to TRUE if
# all listed variables are TRUE
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CBLAS DEFAULT_MSG CBLAS_LIBRARY CBLAS_LIBRARIES)

mark_as_advanced(CBLAS_LIBRARY)

