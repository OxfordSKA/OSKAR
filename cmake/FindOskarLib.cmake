# Find oskar-lib
# Find the oskar-lib includes and library
#
# Defines the following variables:
#
#  OskarLib_FOUND         = True if oskar-lib found
#  OskarLib_LIBRARIES     = Set of libraries required for linking
#  OskarLib_INCLUDE_DIR   = top level header directory.

include(FindPackageHandleStandardArgs)

find_path(OskarLib_INCLUDE_DIR oskar
          PATHS
          /usr/include/
          /usr/local/include/
)

set(OskarLib_NAMES
    oskar_cuda
    oskar_widgets_plotting
)

foreach (lib ${OskarLib_NAMES})
    find_library(OskarLib_${lib}_LIBRARY NAMES ${lib})
    list(APPEND OskarLib_LIBRARIES ${OskarLib_${lib}_LIBRARY})
endforeach ()

find_package_handle_standard_args(OskarLib DEFAULT_MSG
    OskarLib_LIBRARIES OskarLib_INCLUDE_DIR)

#message(STATUS "AAAAAAAAA ${OskarLib_LIBRARIES}")

#if (NOT OskarLib_FOUND)
#    set(OskarLib_LIBRARIES)
#endif ()

#message(STATUS "AAAAAAAAA ${OskarLib_LIBRARIES}")

mark_as_advanced(OskarLib_LIBRARIES OskarLib_INCLUDE_DIR)
