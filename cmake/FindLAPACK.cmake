# - Find a LAPACK library (no includes)
# This module defines
#  LAPACK_LIBRARIES, the libraries needed to use LAPACK.
#  LAPACK_FOUND, If false, do not try to use LAPACK.
# also defined, but not for general use are
#  LAPACK_LIBRARY, where to find the LAPACK library.


set(LAPACK_NAMES ${LAPACK_NAMES} lapack)
find_library(LAPACK_LIBRARY
    NAMES ${LAPACK_NAMES}
    PATHS
    /usr/lib64
    /usr/lib
    /usr/local/lib64
    /usr/local/lib
)


if (LAPACK_LIBRARY)
  set(LAPACK_LIBRARIES ${LAPACK_LIBRARY})
endif (LAPACK_LIBRARY)

# handle the QUIETLY and REQUIRED arguments and set LAPACK_FOUND to TRUE if
# all listed variables are TRUE
include(FindPackageHandleCompat)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(LAPACK DEFAULT_MSG
                                  LAPACK_LIBRARY LAPACK_LIBRARIES)

# Deprecated declarations.
GET_FILENAME_COMPONENT(NATIVE_LAPACK_LIB_PATH ${LAPACK_LIBRARY} PATH)

mark_as_advanced(LAPACK_LIBRARY)
