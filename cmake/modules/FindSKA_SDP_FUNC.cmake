#
# Finds the installed SKA SDP Processing Function Library.
#
# Usage in CMake project:
#
#    find_package(SKA_SDP_FUNC)
#
# If the library is installed in a non-standard location, make sure
# CMAKE_PREFIX_PATH is set appropriately.
#
# This script sets the following CMake variables:
#
#    SKA_SDP_FUNC_FOUND:
#        Boolean flag set to true if the library is found; false if not.
#
#    SKA_SDP_FUNC_INCLUDE_DIRS:
#        Path(s) to include directories.
#
#    SKA_SDP_FUNC_LIBRARIES:
#        Path(s) to library or libraries to link against.
#

include(FindPackageHandleStandardArgs)

find_path(SKA_SDP_FUNC_INCLUDE_DIR ska-sdp-func/sdp_func_version.h
    PATH_SUFFIXES include
)
find_library(SKA_SDP_FUNC_LIBRARY ska_sdp_func PATH_SUFFIXES lib)
if (WIN32)
    find_file(SKA_SDP_FUNC_DLL ska_sdp_func.dll PATH_SUFFIXES bin)
    mark_as_advanced(SKA_SDP_FUNC_DLL)
endif()
mark_as_advanced(SKA_SDP_FUNC_INCLUDE_DIR SKA_SDP_FUNC_LIBRARY)

find_package_handle_standard_args(SKA_SDP_FUNC
    DEFAULT_MSG SKA_SDP_FUNC_LIBRARY SKA_SDP_FUNC_INCLUDE_DIR
)

set(SKA_SDP_FUNC_INCLUDE_DIRS ${SKA_SDP_FUNC_INCLUDE_DIR})
set(SKA_SDP_FUNC_LIBRARIES ${SKA_SDP_FUNC_LIBRARY})
