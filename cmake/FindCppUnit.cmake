# - Find cppunit
# Find the native CPPUNIT includes and library
#
#  CPPUNIT_INCLUDE_DIR - where to find cppunit.h.
#  CPPUNIT_LIBRARIES   - List of libraries when using cppunit.
#  CPPUNIT_FOUND       - True if cppunit found.

# Already in cache, be silent
if (CPPUNIT_INCLUDE_DIR)
    set(CPPUNIT_FIND_QUIETLY TRUE)
endif (CPPUNIT_INCLUDE_DIR)

if (CPPUNIT_INC_DIR)
	find_path(CPPUNIT_INCLUDE_DIR cppunit 
		PATHS ${CPPUNIT_INC_DIR} NO_DEFAULT_PATH)
else()
	find_path(CPPUNIT_INCLUDE_DIR cppunit
    	PATHS
	    /usr/include/
	    /usr/include/cppunit
	    /usr/include/libcppunit
	    ${CPPUNIT_INCLUDES})
endif()

set(CPPUNIT_NAMES cppunit cppunit_dll)

if (CPPUNIT_LIB_DIR)
	foreach( lib ${CPPUNIT_NAMES} )
	    find_library(CPPUNIT_LIBRARY_${lib}  NAMES ${lib} 
	    	PATHS ${CPPUNIT_LIB_DIR} NO_DEFAULT_PATH)
	    if (NOT ${CPPUNIT_LIBRARY_${lib}} MATCHES "CPPUNIT_LIBRARY_${lib}-NOTFOUND")
        	list(APPEND CPPUNIT_LIBRARIES ${CPPUNIT_LIBRARY_${lib}})
    	endif ()
	endforeach(lib)
else()
	foreach(lib ${CPPUNIT_NAMES})
	    find_library(CPPUNIT_LIBRARY_${lib} NAMES ${lib}
	    	PATHS 
	    	/usr/local/lib
	    	/usr/lib
	    	/usr/lib/cppunit
	    	/usr/local/lib/cppunit
         	/usr/local/cppunit/lib)
	    if (NOT ${CPPUNIT_LIBRARY_${lib}} MATCHES "CPPUNIT_LIBRARY_${lib}-NOTFOUND")
        	list(APPEND CPPUNIT_LIBRARIES ${CPPUNIT_LIBRARY_${lib}})
    	endif ()
	endforeach(lib)
endif()

# handle the QUIETLY and REQUIRED arguments and set CPPUNIT_FOUND to TRUE if.
# all listed variables are TRUE
include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CppUnit DEFAULT_MSG CPPUNIT_LIBRARIES CPPUNIT_INCLUDE_DIR)

if(NOT CPPUNIT_FOUND)
    set( CPPUNIT_LIBRARIES )
endif(NOT CPPUNIT_FOUND)

mark_as_advanced(CPPUNIT_LIBRARIES CPPUNIT_INCLUDE_DIR)
