#
# src/interferometry/CMakeLists.txt
#


# ==== Add the source files in this directory.
set(interferometry_SRC
    # CUDA kernels
    cudak/src/oskar_cudak_correlator.cu
    cudak/src/oskar_cudak_correlator_scalar.cu
    cudak/src/oskar_cudak_xyz_to_uvw.cu

    # C/C++
    src/oskar_compute_baselines.c
    src/oskar_cuda_correlator_lm_bw.cu
    src/oskar_cuda_correlator_scalar.cu
    src/oskar_geocentric_cartesian_to_geodetic_spherical.c
    src/oskar_geodetic_spherical_to_geocentric_cartesian.c
    src/oskar_interferometer1_scalar.cu
    src/oskar_horizon_plane_to_geocentric_cartesian.c
    src/oskar_horizon_plane_to_itrs.cpp
    src/oskar_horizon_plane_to_offset_geocentric_cartesian.c
    src/oskar_telescope_model_load_station_coords.cpp
    src/oskar_TelescopeModel.cu
    src/oskar_VisData.c  # deprecated
    src/oskar_visibilities_append.cpp
    src/oskar_visibilities_init.cpp
    src/oskar_visibilities_insert.cpp
    src/oskar_visibilities_read.cpp
    src/oskar_visibilities_resize.cpp
    src/oskar_visibilities_write.cpp
    src/oskar_Visibilities.cpp
    src/oskar_xyz_to_uvw.c
)
set(interferometry_SRC "${interferometry_SRC}" PARENT_SCOPE)

add_subdirectory(test)
add_subdirectory(matlab)
