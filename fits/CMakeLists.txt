#
# src/fits/CMakeLists.txt
#

set(fits_SRC
    src/oskar_fits_check_status.c
    src/oskar_fits_image_write.c
    src/oskar_fits_write_axis_header.c
    src/oskar_fits_write.c
    #src/oskar_fits_write_cube.c
    src/oskar_uvfits_writer.c # Needs fixing for the new visibility structure.
)
set(fits_SRC "${fits_SRC}" PARENT_SCOPE)

# ==== Macros to build the FITS module.
include_directories(${CFITSIO_INCLUDE_DIR})
add_library(oskar_fits ${fits_SRC})
target_link_libraries(oskar_fits ${CFITSIO_LIBRARIES})

set_target_properties(oskar_fits
    PROPERTIES
    #PREFIX "lib"
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
    INSTALL_RPATH_USE_LINK_PATH TRUE
    SOVERSION ${oskar-lib_SOVERSION})



# ==== Install configuration for the FITS module.
install(TARGETS oskar_fits DESTINATION ${OSKAR_LIB_INSTALL_DIR})
set_target_properties(oskar_fits PROPERTIES SOVERSION ${oskar-lib_SOVERSION})

# ==== Recurse into test subdirectory.
add_subdirectory(test)

