#
# oskar/sky/CMakeLists.txt
#

set(sky_SRC
    define_sky_copy_source_data.h
    define_sky_scale_flux_with_frequency.h
    define_update_horizon_mask.h
    #src/oskar_evaluate_tec_tid.c
    src/oskar_generate_random_coordinate.c
    src/oskar_sky_accessors.c
    src/oskar_sky_append_to_set.c
    src/oskar_sky_append.c
    src/oskar_sky_copy.c
    src/oskar_sky_copy_contents.c
    src/oskar_sky_copy_source_data.c
    src/oskar_sky_create.c
    src/oskar_sky_create_copy.c
    src/oskar_sky_evaluate_gaussian_source_parameters.c
    src/oskar_sky_evaluate_relative_directions.c
    src/oskar_sky_filter_by_flux.c
    src/oskar_sky_filter_by_radius.c
    src/oskar_sky_from_fits_file.c
    src/oskar_sky_from_healpix_ring.c
    src/oskar_sky_from_image.c
    src/oskar_sky_free.c
    src/oskar_sky_generate_grid.c
    src/oskar_sky_generate_random_power_law.c
    src/oskar_sky_horizon_clip.c
    src/oskar_sky_load.c
    src/oskar_sky_override_polarisation.c
    src/oskar_sky_read.c
    src/oskar_sky_resize.c
    #src/oskar_sky_rotate_to_position.c
    src/oskar_sky_save.c
    src/oskar_sky_scale_flux_with_frequency.c
    src/oskar_sky_set_gaussian_parameters.c
    src/oskar_sky_set_source.c
    src/oskar_sky_set_spectral_index.c
    src/oskar_sky_write.c
    src/oskar_sky.cl
    src/oskar_update_horizon_mask.c
)

if (CUDA_FOUND)
    list(APPEND sky_SRC src/oskar_sky.cu)
endif()

set(sky_SRC "${sky_SRC}" PARENT_SCOPE)

# Build tests.
if (BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
    add_subdirectory(test)
endif()
