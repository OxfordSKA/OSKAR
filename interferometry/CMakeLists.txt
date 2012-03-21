#
# src/interferometry/CMakeLists.txt
#

# Add the source files in this directory.
set(interferometry_SRC
    src/oskar_compute_baselines.c
    src/oskar_correlate.cu
    src/oskar_cuda_correlator_lm_bw.cu # Deprecated.
    src/oskar_evaluate_baseline_noise_stddev.c
    src/oskar_evaluate_baseline_uvw.cpp
    src/oskar_evaluate_baselines.cpp
    src/oskar_evaluate_jones_K.cu
    src/oskar_evaluate_station_uvw.cpp
    src/oskar_geocentric_cartesian_to_geodetic_spherical.c
    src/oskar_geodetic_spherical_to_geocentric_cartesian.c
    src/oskar_horizon_plane_to_geocentric_cartesian.c
    src/oskar_horizon_plane_to_offset_geocentric_cartesian.c
    src/oskar_interferometer_scalar.cpp
    src/oskar_interferometer.cpp
    src/oskar_offset_geocentric_cartesian_to_geocentric_cartesian.c
    src/oskar_telescope_model_analyse.c
    src/oskar_telescope_model_copy.c
    src/oskar_telescope_model_free.c
    src/oskar_telescope_model_init.c
    src/oskar_telescope_model_load_station_coords.c
    src/oskar_telescope_model_location.c
    src/oskar_telescope_model_multiply_by_wavenumber.c
    src/oskar_telescope_model_resize.c
    src/oskar_telescope_model_scale_coords.c
    src/oskar_telescope_model_set_station_coords.c
    src/oskar_telescope_model_type.c
    src/oskar_TelescopeModel.cpp
    src/oskar_visibilities_add_sky_noise.c
    src/oskar_visibilities_evaluate_sky_noise_stddev.c
    src/oskar_visibilities_clear_contents.c
    src/oskar_visibilities_copy.c
    src/oskar_visibilities_free.c
    src/oskar_visibilities_get_channel_amps.c
    src/oskar_visibilities_init.c
    src/oskar_visibilities_location.c
    src/oskar_visibilities_read.c
    src/oskar_visibilities_resize.c
    src/oskar_visibilities_write.c
    src/oskar_Visibilities.cpp
    src/oskar_xyz_to_uvw_cuda.cu
    src/oskar_xyz_to_uvw.c
)

# Add the CUDA kernels.
add_subdirectory(cudak)
foreach (file ${interferometry_cuda_kernels_SRC})
    list(APPEND interferometry_SRC cudak/${file})
endforeach ()

set(interferometry_SRC "${interferometry_SRC}" PARENT_SCOPE)

add_subdirectory(test)
add_subdirectory(matlab)
