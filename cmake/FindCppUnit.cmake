# - Find cppunit
# Find the native CPPUNIT includes and library
#
#  CPPUNIT_INCLUDE_DIR - where to find cppunit.h.
#  CPPUNIT_LIBRARIES   - List of libraries when using cppunit.
#  CPPUNIT_FOUND       - True if cppunit found.

# Already in cache, be silent
IF (CPPUNIT_INCLUDE_DIR)
    SET(CPPUNIT_FIND_QUIETLY TRUE)
ENDIF (CPPUNIT_INCLUDE_DIR)

FIND_PATH(CPPUNIT_INCLUDE_DIR TestSuite.h PATHS /usr/include/cppunit /usr/include/libcppunit )

SET(CPPUNIT_NAMES cppunit)
FOREACH( lib ${CPPUNIT_NAMES} )
    FIND_LIBRARY(CPPUNIT_LIBRARY_${lib} NAMES ${lib} )
    LIST(APPEND CPPUNIT_LIBRARIES ${CPPUNIT_LIBRARY_${lib}})
ENDFOREACH(lib)

# handle the QUIETLY and REQUIRED arguments and set CPPUNIT_FOUND to TRUE if.
# all listed variables are TRUE
INCLUDE(FindPackageHandleCompat)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CppUnit DEFAULT_MSG CPPUNIT_LIBRARIES CPPUNIT_INCLUDE_DIR)

IF(NOT CPPUNIT_FOUND)
    SET( CPPUNIT_LIBRARIES )
ENDIF(NOT CPPUNIT_FOUND)

MARK_AS_ADVANCED(CPPUNIT_LIBRARIES CPPUNIT_INCLUDE_DIR)

