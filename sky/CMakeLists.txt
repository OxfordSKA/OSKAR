#
# src/sky/CMakeLists.txt
#

include_directories(${oskar-lib_SOURCE_DIR}/math)

# Add the source files in this directory.
set(sky_SRC
    src/oskar_date_time_to_mjd.c
    src/oskar_equation_of_equinoxes_fast.c
    src/oskar_evaluate_gaussian_source_parameters.c
    src/oskar_evaluate_jones_R.cu
    src/oskar_evaluate_sky_temperature.c
    src/oskar_galactic_to_fk5.c
    src/oskar_generate_random_coordinate.c
    src/oskar_ha_dec_to_az_el.c
    src/oskar_ha_dec_to_hor_lmn.c
    src/oskar_lm_to_n.c
    src/oskar_mjd_to_gast_fast.c
    src/oskar_mjd_to_gmst.c
    src/oskar_mjd_to_last_fast.c
    src/oskar_mjd_to_lmst.c
    src/oskar_ra_dec_to_az_el_cuda.cu
    src/oskar_ra_dec_to_hor_lmn_cuda.cu
    src/oskar_ra_dec_to_hor_lmn.c
    src/oskar_ra_dec_to_rel_lmn_cuda.cu
    src/oskar_ra_dec_to_rel_lmn.c
    src/oskar_sky_model_append_to_set.c
    src/oskar_sky_model_append.c
    src/oskar_sky_model_combine_set.c
    src/oskar_sky_model_compute_relative_lmn.c
    src/oskar_sky_model_copy.c
    src/oskar_sky_model_filter_by_flux.c
    src/oskar_sky_model_filter_by_radius.c
    src/oskar_sky_model_free.c
    src/oskar_sky_model_get_ptr.c
    src/oskar_sky_model_horizon_clip.cu
    src/oskar_sky_model_init.c
    src/oskar_sky_model_insert.c
    src/oskar_sky_model_load.c
    src/oskar_sky_model_load_gsm.c
    src/oskar_sky_model_location.c
    src/oskar_sky_model_resize.c
    src/oskar_sky_model_rotate_to_position.c
    src/oskar_sky_model_scale_by_spectral_index.cu
    src/oskar_sky_model_set_gaussian_parameters.c
    src/oskar_sky_model_set_source.c
    src/oskar_sky_model_split.c
    src/oskar_sky_model_type.c
    src/oskar_sky_model_write.c
    src/oskar_SkyModel.cpp
)

# Add the CUDA kernels.
add_subdirectory(cudak)
foreach (file ${sky_cuda_kernels_SRC})
    list(APPEND sky_SRC cudak/${file})
endforeach ()

set(sky_SRC "${sky_SRC}" PARENT_SCOPE)

# Build MATLAB mex functions.
add_subdirectory(matlab)

# Build tests.
add_subdirectory(test)
