#
# src/fits/CMakeLists.txt
#

if (CFITSIO_FOUND AND CUDA_FOUND)

    include_directories(${CFITSIO_INCLUDE_DIR})

    # Fits source code.
    set(fits_SRC
        src/oskar_fits_check_status.c
        src/oskar_fits_image_write.c
        src/oskar_fits_to_sky_model.c
        src/oskar_fits_healpix_to_sky_model.c
        src/oskar_fits_write_axis_header.c
        src/oskar_fits_write.c
        #src/oskar_uvfits_writer.c
    )
    set(fits_SRC "${fits_SRC}" PARENT_SCOPE)

    # Build and install the oskar_fits library.
    include_directories(${CFITSIO_INCLUDE_DIR})
    set(libname oskar_fits)
    add_library(${libname} ${fits_SRC})
    target_link_libraries(${libname} ${CFITSIO_LIBRARIES} oskar)
    set_target_properties(${libname} PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
        INSTALL_RPATH_USE_LINK_PATH TRUE
        SOVERSION ${OSKAR_VERSION})
    install(TARGETS ${libname} DESTINATION ${OSKAR_LIB_INSTALL_DIR})

    # Unit tests.
    add_subdirectory(test)

endif (CFITSIO_FOUND AND CUDA_FOUND)
