#
# src/station/CMakeLists.txt
#

set(station_SRC
    src/oskar_beamPattern.cpp
    #src/oskar_bf_cuda_ant_sig_iso_2d.cu
    src/oskar_element_model_copy_to_gpu.c
    src/oskar_element_model_evaluate.c
    src/oskar_element_model_free_gpu.c
    src/oskar_element_model_free.c
    src/oskar_element_model_load.c
    src/oskar_evaluate_beam_horizontal_lmn.c
    src/oskar_evaluate_jones_E.cpp
    src/oskar_evaluate_source_horizontal_lmn.cpp
    src/oskar_evaluate_station_beam_scalar.cu
    src/oskar_evaluate_station_beam.cu # probably doesn't need to be cu soon.
    src/oskar_phi_theta_to_normalised_cuda.cu
    src/oskar_station_model_copy.c
    src/oskar_station_model_free.c
    src/oskar_station_model_init.c
    src/oskar_station_model_load.c
    src/oskar_station_model_location.c
    src/oskar_station_model_multiply_by_wavenumber.c
    src/oskar_station_model_resize.c
    src/oskar_station_model_scale_coords.c
    src/oskar_station_model_set_element_errors.c
    src/oskar_station_model_set_element_pos.c
    src/oskar_station_model_set_element_weight.c
    src/oskar_station_model_type.c
    src/oskar_station_model_write_coords.c
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
