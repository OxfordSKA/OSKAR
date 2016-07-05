#! CMake initialisation.
cmake_minimum_required(VERSION 2.8.3)
if (POLICY CMP0042)
    cmake_policy(SET CMP0042 OLD)
endif()
if (POLICY CMP0045)
    cmake_policy(SET CMP0045 OLD)
endif()

#! Declare the CMake project.
project(oskar_imager C)

#! Set the name of the library.
set(libname ${PROJECT_NAME})

# Add the source files.
set(${libname}_SRC
    src/oskar_fftpack_cfft.c
    src/oskar_fftpack_cfft_f.c
    src/oskar_fftphase.c
    src/oskar_grid_simple.c
    src/oskar_grid_weights.c
    src/oskar_grid_wproj.c
    src/oskar_grid_correction.c
    src/oskar_grid_functions_spheroidal.c
    src/oskar_grid_functions_pillbox.c
    src/oskar_imager_accessors.c
    src/oskar_imager_check_init.c
    src/oskar_imager_create.c
    src/oskar_imager_finalise.c
    src/oskar_imager_free.c
    src/oskar_imager_linear_to_stokes.c
    src/oskar_imager_reset_cache.c
    src/oskar_imager_rotate_coords.c
    src/oskar_imager_rotate_vis.c
    src/oskar_imager_run.c
    src/oskar_imager_select_coords.c
    src/oskar_imager_select_vis.c
    src/oskar_imager_update.c
    src/private_imager_create_fits_files.c
    src/private_imager_free_dft.c
    src/private_imager_free_fft.c
    src/private_imager_free_gpu_data.c
    src/private_imager_free_wproj.c
    src/private_imager_init_dft.c
    src/private_imager_init_fft.c
    src/private_imager_init_wproj.c
    src/private_imager_set_num_planes.c
    src/private_imager_update_plane_dft.c
    src/private_imager_update_plane_fft.c
    src/private_imager_update_plane_wproj.c
    src/private_imager_weight_radial.c
    src/private_imager_weight_uniform.c
)

add_library(${libname} ${${libname}_SRC})
target_link_libraries(${libname}
    cfitsio
    oskar_log
    oskar_binary
    oskar)

if (CASACORE_FOUND)
    target_link_libraries(${libname} oskar_ms)
endif()

if (CUDA_FOUND)
    target_link_libraries(${libname} ${CUDA_CUFFT_LIBRARIES})
endif()

if (MSVC)
    set_target_properties(${libname} PROPERTIES
        COMPILE_FLAGS "${OpenMP_C_FLAGS}"
        SOVERSION ${OSKAR_VERSION})
else ()
    set_target_properties(${libname} PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
        INSTALL_RPATH_USE_LINK_PATH TRUE
        COMPILE_FLAGS "${OpenMP_C_FLAGS}"
        LINK_FLAGS "${OpenMP_C_FLAGS}"
        SOVERSION ${OSKAR_VERSION}
        VERSION ${OSKAR_VERSION})
endif()

install(TARGETS ${libname} DESTINATION ${OSKAR_LIB_INSTALL_DIR})

add_subdirectory(test)

