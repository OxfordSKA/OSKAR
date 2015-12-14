#
# src/fits/CMakeLists.txt
#

if (CUDA_FOUND)

    # Fits source code.
    set(fits_SRC
        src/oskar_fits_check_status.c
        src/oskar_fits_healpix_write_image.c
        src/oskar_fits_healpix_write_mem.c
        src/oskar_fits_image_write.c
        src/oskar_fits_image_to_sky_model.c
        src/oskar_fits_healpix_to_sky_model.c
        src/oskar_fits_write_axis_header.c
        src/oskar_fits_write.c
    )
    set(fits_SRC "${fits_SRC}" PARENT_SCOPE)

    # Build and install the oskar_fits library.
    set(libname oskar_fits)
    add_library(${libname} ${fits_SRC})
    target_link_libraries(${libname} cfitsio oskar)
    if (APPLE)
        set_target_properties(${libname} PROPERTIES
            INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
            INSTALL_RPATH_USE_LINK_PATH TRUE
            SOVERSION ${OSKAR_VERSION}
            VERSION ${OSKAR_VERSION})
    else()
        set_target_properties(${libname} PROPERTIES
            INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
            INSTALL_RPATH_USE_LINK_PATH TRUE
            SOVERSION ${OSKAR_VERSION}
            VERSION ${OSKAR_VERSION})
    endif()
    install(TARGETS ${libname} DESTINATION ${OSKAR_LIB_INSTALL_DIR})

    # Unit tests.
    add_subdirectory(test)

endif (CUDA_FOUND)
