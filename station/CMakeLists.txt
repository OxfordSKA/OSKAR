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
    src/oskar_mult_beampattern_by_sources.cu
    src/oskar_StationModel.cu
)
set(station_SRC "${station_SRC}" PARENT_SCOPE)


# ==== Recurse into test subdirectory.
add_subdirectory(test)
