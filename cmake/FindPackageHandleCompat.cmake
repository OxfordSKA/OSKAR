#
# Wrapper to FindPackageHandleStandardArgs required for message printing
# in find_package macros to fix compatibility with cmake before 2.5
#
IF("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" GREATER 2.5 )
    INCLUDE(FindPackageHandleStandardArgs)
ELSE("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" GREATER 2.5)
    MACRO(FIND_PACKAGE_HANDLE_STANDARD_ARGS package)
        IF(${package}_LIBRARY AND ${package}_INCLUDE_DIR)
            SET(${package}_FOUND TRUE)
        ENDIF(${package}_LIBRARY AND ${package}_INCLUDE_DIR)
    ENDMACRO(FIND_PACKAGE_HANDLE_STANDARD_ARGS package)
ENDIF("${CMAKE_MAJOR_VERSION}.${CMAKE_MINOR_VERSION}" GREATER 2.5)
