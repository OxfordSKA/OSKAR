#
# src/fits/CMakeLists.txt
#

set(fits_SRC
    src/oskar_uvfits_writer.c # Needs fixing for the new visibility structure.
)
set(fits_SRC "${fits_SRC}" PARENT_SCOPE)

# ==== Macros to build the oskar-lib ms module.
include_directories(${CFITSIO_INCLUDE_DIR})
include_directories(${QT_INCLUDE_DIR})
add_library(oskar_fits ${fits_SRC})
set_target_properties(oskar_fits PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
target_link_libraries(oskar_fits
    ${CFITSIO_LIBRARIES}
    ${QT_QTCORE_LIBRARY}
)

# ==== Install configuration for the oskar-lib ms module.
install(TARGETS oskar_fits DESTINATION ${OSKAR_LIB_INSTALL_DIR})
set_target_properties(oskar_fits PROPERTIES SOVERSION ${oskar-lib_SOVERSION})

# ==== Recurse into test subdirectory.
add_subdirectory(test)

