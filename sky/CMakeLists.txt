#
# src/sky/CMakeLists.txt
#

# Add the source files in this directory.
set(sky_SRC
    src/oskar_date_time_to_mjd.c
    src/oskar_equation_of_equinoxes_fast.c
    src/oskar_evaluate_jones_R_cuda.cu
    src/oskar_evaluate_jones_R.c
    src/oskar_evaluate_jones_Z.c
    src/oskar_evaluate_source_radius.c
    src/oskar_evaluate_TEC_TID.c
    src/oskar_galactic_to_fk5.c
    src/oskar_generate_random_coordinate.c
    src/oskar_ha_dec_to_az_el.c
    src/oskar_ha_dec_to_hor_lmn.c
    src/oskar_lm_to_n.c
    src/oskar_load_TID_parameter_file.c
    src/oskar_mjd_to_gast_fast.c
    src/oskar_mjd_to_gmst.c
    src/oskar_mjd_to_last_fast.c
    src/oskar_mjd_to_lmst.c
    src/oskar_ra_dec_to_az_el_cuda.cu
    src/oskar_ra_dec_to_hor_lmn_cuda.cu
    src/oskar_ra_dec_to_hor_lmn.c
    src/oskar_ra_dec_to_rel_lmn_cuda.cu
    src/oskar_ra_dec_to_rel_lmn.c
    #src/oskar_rebin_sky_cuda.cu # Doesn't work on compute 1.3 architectures.
    src/oskar_scale_flux_by_spectral_index_cuda.cu
    src/oskar_scale_flux_by_spectral_index.c
    src/oskar_sky_accessors.c
    src/oskar_sky_append_to_set.c
    src/oskar_sky_append.c
    src/oskar_sky_combine_set.c
    src/oskar_sky_compute_relative_lmn.c
    src/oskar_sky_compute_source_radius.c
    src/oskar_sky_create.c
    src/oskar_sky_create_copy.c
    src/oskar_sky_evaluate_gaussian_source_parameters.c
    src/oskar_sky_filter_by_flux.c
    src/oskar_sky_filter_by_flux_cuda.cu
    src/oskar_sky_filter_by_radius.c
    src/oskar_sky_free.c
    src/oskar_sky_get_ptr.c
    src/oskar_sky_horizon_clip.cu
    src/oskar_sky_init.c
    src/oskar_sky_insert.c
    src/oskar_sky_load.c
    src/oskar_sky_load_gsm.c
    src/oskar_sky_read.c
    src/oskar_sky_resize.c
    src/oskar_sky_rotate_to_position.c
    src/oskar_sky_save.c
    src/oskar_sky_scale_by_spectral_index.c
    src/oskar_sky_set_gaussian_parameters.c
    src/oskar_sky_set_source.c
    src/oskar_sky_set_spectral_index.c
    src/oskar_sky_write.c
    src/oskar_WorkJonesZ.c
    src/oskar_update_horizon_mask_cuda.cu
    src/oskar_update_horizon_mask.c
)

# Add the CUDA kernels.
add_subdirectory(cudak)
foreach (file ${sky_cuda_kernels_SRC})
    list(APPEND sky_SRC cudak/${file})
endforeach ()

set(sky_SRC "${sky_SRC}" PARENT_SCOPE)

# Build tests.
add_subdirectory(test)
