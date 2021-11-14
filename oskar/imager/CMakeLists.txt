#
# oskar/imager/CMakeLists.txt
#

set(imager_SRC
    define_grid_correction.h
    define_grid_tile_grid.h
    define_grid_tile_utils.h
    define_imager_generate_w_phase_screen.h
    src/oskar_grid_correction.c
    src/oskar_grid_functions_spheroidal.c
    src/oskar_grid_functions_pillbox.c
    src/oskar_grid_simple.c
    src/oskar_grid_weights.c
    #src/oskar_grid_wproj.c
    src/oskar_grid_wproj2.c
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
    src/oskar_imager_update.c
    src/oskar_imager_gpu.cl
    src/oskar_imager.cl
    src/private_imager_composite_nearest_even.c
    src/private_imager_create_fits_files.c
    src/private_imager_filter_time.c
    src/private_imager_filter_uv.c
    src/private_imager_free_device_data.c
    src/private_imager_generate_w_phase_screen.c
    src/private_imager_init_dft.c
    src/private_imager_init_fft.c
    src/private_imager_init_wproj.c
    src/private_imager_read_coords.c
    src/private_imager_read_data.c
    src/private_imager_read_dims.c
    src/private_imager_select_data.c
    src/private_imager_set_num_planes.c
    src/private_imager_taper_weights.c
    src/private_imager_update_plane_dft.c
    src/private_imager_update_plane_fft.c
    src/private_imager_update_plane_wproj.c
    src/private_imager_weight_radial.c
    src/private_imager_weight_uniform.c
)

if (CUDA_FOUND)
    list(APPEND imager_SRC src/oskar_imager.cu)
endif()

set(imager_SRC "${imager_SRC}" PARENT_SCOPE)

# Build tests.
if (BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
    add_subdirectory(test)
endif()

