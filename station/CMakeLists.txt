#
# src/station/CMakeLists.txt
#

set(station_SRC
    src/oskar_blank_below_horizon.cu
    src/oskar_element_model_copy.c
    src/oskar_element_model_evaluate.cu
    src/oskar_element_model_free.c
    src/oskar_element_model_init.c
    src/oskar_element_model_load_cst.c
    src/oskar_element_model_type.c
    src/oskar_ElementModel.cpp
    src/oskar_evaluate_beam_horizontal_lmn.c
    src/oskar_evaluate_dipole_pattern.cu
    src/oskar_evaluate_jones_E.c
    src/oskar_evaluate_source_horizontal_lmn.c
    src/oskar_evaluate_effective_area.c
    src/oskar_evaluate_element_weights_errors.cu
    src/oskar_evaluate_flux_density.c
    src/oskar_evaluate_spline_pattern.cu
    src/oskar_evaluate_station_beam_dipoles.cu
    src/oskar_evaluate_station_beam_scalar.cu
    src/oskar_evaluate_station_beam.c
    src/oskar_evaluate_station_receiver_noise_stddev.c
    src/oskar_evaluate_tapered_dipole_pattern.cu
    src/oskar_station_model_analyse.c
    src/oskar_station_model_copy.c
    src/oskar_station_model_free.c
    src/oskar_station_model_init.c
    src/oskar_station_model_load_config.c
    src/oskar_station_model_location.c
    src/oskar_station_model_multiply_by_wavenumber.c
    src/oskar_station_model_resize.c
    src/oskar_station_model_save_config.c
    src/oskar_station_model_scale_coords.c
    src/oskar_station_model_set_element_coords.c
    src/oskar_station_model_set_element_errors.c
    src/oskar_station_model_set_element_orientation.c
    src/oskar_station_model_set_element_weight.c
    src/oskar_station_model_type.c
    src/oskar_StationModel.cpp
)

# Add the CUDA kernels.
add_subdirectory(cudak)
foreach (file ${station_cuda_kernels_SRC})
    list(APPEND station_SRC cudak/${file})
endforeach ()

set(station_SRC "${station_SRC}" PARENT_SCOPE)

# ==== Recurse into test subdirectory.
add_subdirectory(test)
