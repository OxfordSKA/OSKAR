#
# src/sky/CMakeLists.txt
#

# Add the source files in this directory.
set(sky_SRC
    src/oskar_evaluate_tec_tid.c
    src/oskar_generate_random_coordinate.c
    src/oskar_scale_flux_with_frequency.c
    src/oskar_sky_accessors.c
    src/oskar_sky_append_to_set.c
    src/oskar_sky_append.c
    src/oskar_sky_combine_set.c
    src/oskar_sky_copy.c
    src/oskar_sky_copy_contents.c
    #src/oskar_sky_copy_filter_bands.c
    src/oskar_sky_copy_source_data.c
    src/oskar_sky_create.c
    src/oskar_sky_create_copy.c
    src/oskar_sky_evaluate_gaussian_source_parameters.c
    src/oskar_sky_evaluate_relative_directions.c
    src/oskar_sky_filter_by_flux.c
    src/oskar_sky_filter_by_radius.c
    src/oskar_sky_free.c
    src/oskar_sky_horizon_clip.c
    src/oskar_sky_load.c
    src/oskar_sky_load_gsm.c
    src/oskar_sky_override_polarisation.c
    src/oskar_sky_read.c
    src/oskar_sky_resize.c
    src/oskar_sky_rotate_to_position.c
    src/oskar_sky_save.c
    src/oskar_sky_scale_flux_with_frequency.c
    #src/oskar_sky_set_filter_bands.c
    src/oskar_sky_set_gaussian_parameters.c
    src/oskar_sky_set_source.c
    src/oskar_sky_set_spectral_index.c
    src/oskar_sky_write.c
    src/oskar_update_horizon_mask.c
)

if (CUDA_FOUND)
    list(APPEND sky_SRC
        src/oskar_scale_flux_with_frequency_cuda.cu
        #src/oskar_rebin_sky_cuda.cu # Doesn't work on compute 1.3 architectures.
        src/oskar_sky_copy_source_data_cuda.cu
        src/oskar_sky_filter_by_flux_cuda.cu
        src/oskar_update_horizon_mask_cuda.cu
    )
endif()

set(sky_SRC "${sky_SRC}" PARENT_SCOPE)

# Build tests.
add_subdirectory(test)
