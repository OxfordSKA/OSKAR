#
# src/station/CMakeLists.txt
#

set(station_SRC
    src/oskar_apply_element_taper_cosine.c
    src/oskar_apply_element_taper_gaussian.c
    src/oskar_blank_below_horizon.c
    src/oskar_element_accessors.c
    src/oskar_element_copy.c
    src/oskar_element_create.c
    src/oskar_element_evaluate.c
    src/oskar_element_free.c
    src/oskar_element_load_cst.c
    src/oskar_element_read.c
    src/oskar_element_resize_frequency_data.c
    src/oskar_element_write.c
    src/oskar_evaluate_array_pattern.c
    src/oskar_evaluate_array_pattern_dipoles.c
    src/oskar_evaluate_array_pattern_hierarchical.c
    src/oskar_evaluate_beam_horizon_direction.c
    src/oskar_evaluate_dipole_pattern.c
    src/oskar_evaluate_geometric_dipole_pattern.c
    src/oskar_evaluate_jones_E.c
    src/oskar_evaluate_pierce_points.c
    src/oskar_evaluate_element_weights_dft.c
    src/oskar_evaluate_element_weights_errors.c
    src/oskar_evaluate_element_weights.c
    src/oskar_evaluate_station_beam_aperture_array.c
    src/oskar_evaluate_station_beam_gaussian.c
    src/oskar_evaluate_station_beam_pattern.c
    src/oskar_evaluate_station_from_telescope_dipole_azimuth.c
    src/oskar_evaluate_vla_beam_pbcor.c
    src/oskar_station_accessors.c
    src/oskar_station_analyse.c
    src/oskar_station_create_child_stations.c
    src/oskar_station_create_copy.c
    src/oskar_station_create.c
    src/oskar_station_different.c
    src/oskar_station_duplicate_first_child.c
    src/oskar_station_free.c
    src/oskar_station_load_apodisation.c
    src/oskar_station_load_config.c
    src/oskar_station_load_gain_phase.c
    src/oskar_station_load_layout.c
    src/oskar_station_load_orientation.c
    src/oskar_station_load_permitted_beams.c
    src/oskar_station_override_element_gains.c
    src/oskar_station_override_element_orientations.c
    src/oskar_station_override_element_phases.c
    src/oskar_station_override_element_time_variable_gains.c
    src/oskar_station_override_element_time_variable_phases.c
    src/oskar_station_override_element_xy_position_errors.c
    src/oskar_station_resize_element_types.c
    src/oskar_station_resize.c
    src/oskar_station_save_config.c
    src/oskar_station_set_element_coords.c
    src/oskar_station_set_element_errors.c
    src/oskar_station_set_element_orientation.c
    src/oskar_station_set_element_weight.c
    src/oskar_system_noise_model.c
    src/oskar_system_noise_model_copy.c
    src/oskar_system_noise_model_create.c
    src/oskar_system_noise_model_free.c
    src/oskar_system_noise_model_load.c
    src/oskar_station_work.c
)

if (CUDA_FOUND)
    list(APPEND station_SRC
        src/oskar_apply_element_taper_cosine_cuda.cu
        src/oskar_apply_element_taper_gaussian_cuda.cu
        src/oskar_blank_below_horizon_cuda.cu
        src/oskar_evaluate_array_pattern_dipoles_cuda.cu
        src/oskar_evaluate_dipole_pattern_cuda.cu
        src/oskar_evaluate_geometric_dipole_pattern_cuda.cu
        src/oskar_evaluate_element_weights_errors_cuda.cu
        src/oskar_evaluate_element_weights_dft_cuda.cu
        src/oskar_evaluate_vla_beam_pbcor_cuda.cu
    )
endif()

set(station_SRC "${station_SRC}" PARENT_SCOPE)

# ==== Recurse into test subdirectory.
add_subdirectory(test)
