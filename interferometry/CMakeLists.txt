#
# src/interferometry/CMakeLists.txt
#

# Add the source files in this directory.
set(interferometry_SRC
    src/oskar_correlate_gaussian_omp.c
    src/oskar_correlate_gaussian_time_smearing_omp.c
    src/oskar_correlate_point_omp.c
    src/oskar_correlate_point_time_smearing_omp.c
    src/oskar_correlate.c

    src/oskar_telescope_accessors.c
    src/oskar_telescope_analyse.c
    src/oskar_telescope_config_override.c
    src/oskar_telescope_create.c
    src/oskar_telescope_create_copy.c
    src/oskar_telescope_duplicate_first_station.c
    src/oskar_telescope_free.c
    src/oskar_telescope_load_pointing_file.c
    src/oskar_telescope_load_station_coords_ecef.c
    src/oskar_telescope_load_station_coords_horizon.c
    src/oskar_telescope_resize.c
    src/oskar_telescope_save_station_coords.c
    src/oskar_telescope_set_station_coords.c

    src/oskar_vis_accessors.c
    src/oskar_vis_add_system_noise.c
    src/oskar_vis_create.c
    src/oskar_vis_create_copy.c
    src/oskar_vis_free.c
    src/oskar_vis_get_channel_amps.c
    src/oskar_vis_read.c
    src/oskar_vis_resize.c
    #src/oskar_vis_rotate_polarisation.c
    src/oskar_vis_write.c
)

if (CUDA_FOUND)
    list(APPEND interferometry_SRC
        src/oskar_correlate_gaussian_cuda.cu
        src/oskar_correlate_gaussian_time_smearing_cuda.cu
        src/oskar_correlate_point_cuda.cu
        src/oskar_correlate_point_time_smearing_cuda.cu
        src/oskar_correlate_scalar_point_cuda.cu
    )
endif()

set(interferometry_SRC "${interferometry_SRC}" PARENT_SCOPE)

add_subdirectory(test)
