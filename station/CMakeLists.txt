#
# src/station/CMakeLists.txt
#

set(station_SRC
    src/oskar_beamPattern.cpp
    #src/oskar_bf_cuda_ant_sig_iso_2d.cu
    src/oskar_evaluate_station_beam.cu
    src/oskar_element_model_copy_to_gpu.cu
    src/oskar_element_model_free_gpu.cu
    src/oskar_element_model_free.c
    src/oskar_element_model_load.c
    src/oskar_evaluate_jones_E.cpp
    src/oskar_station_model_copy.c
    src/oskar_station_model_init.c
    src/oskar_station_model_load_coords.cpp
    src/oskar_station_model_resize.cpp
    src/oskar_StationModel.cu
)

# Add the CUDA kernels.
add_subdirectory(cudak)
foreach (file ${station_cuda_kernels_SRC})
    list(APPEND station_SRC cudak/${file})
endforeach ()

set(station_SRC "${station_SRC}" PARENT_SCOPE)

# ==== Recurse into test subdirectory.
add_subdirectory(test)
