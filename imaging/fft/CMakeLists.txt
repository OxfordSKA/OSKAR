#
# src/imaging/fft/CMakeLists.txt
#
set(BUILD_OSKAR_IMAGING OFF)

if (BUILD_OSKAR_IMAGING)

    # ==== Create list of source files for the oskar-lib imaging module.
    set(imaging_SRC
        src/oskar_evaluate_gridding_kernels.c
        #src/oskar_fft_imager.c
        src/oskar_fft_utility.c
        src/oskar_gridding.c # Needs fix for new visibility structure.
        src/oskar_grid_correct.c
        src/oskar_VisGrid.c
    )
    #set(imaging_SRC "${imaging_SRC}" PARENT_SCOPE)

    # ==== Build the oskar-lib imaging module.
    add_library(oskar_imaging ${imaging_SRC})
    target_link_libraries(oskar_imaging
        ${FFTW3_FFTWF_LIBRARY}
        ${FFTW3_FFTW_LIBRARY})
    set_target_properties(oskar_imaging
        PROPERTIES
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        LINK_FLAGS "${OpenMP_CXX_FLAGS}"
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
        INSTALL_RPATH_USE_LINK_PATH TRUE
        SOVERSION ${oskar-lib_SOVERSION})

    #install(TARGETS oskar_imaging LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR})

    # ==== Recurse into test subdirectory.
    add_subdirectory(test)

endif (BUILD_OSKAR_IMAGING)
