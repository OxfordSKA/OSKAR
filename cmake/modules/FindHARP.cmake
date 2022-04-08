include(FindPackageHandleStandardArgs)

find_path(HARP_INCLUDE_DIR harp_beam.h PATH_SUFFIXES include)
find_library(HARP_LIBRARY harp_beam PATH_SUFFIXES lib)
if (WIN32)
    find_file(HARP_DLL harp_beam.dll PATH_SUFFIXES bin)
    mark_as_advanced(HARP_DLL)
endif()
mark_as_advanced(HARP_INCLUDE_DIR HARP_LIBRARY)

find_package_handle_standard_args(HARP
    DEFAULT_MSG HARP_LIBRARY HARP_INCLUDE_DIR)

set(HARP_INCLUDE_DIRS ${HARP_INCLUDE_DIR})
set(HARP_LIBRARIES ${HARP_LIBRARY})
