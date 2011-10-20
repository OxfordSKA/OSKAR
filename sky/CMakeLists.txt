#
# src/sky/CMakeLists.txt
#

include_directories(${oskar-lib_SOURCE_DIR}/math)

# Add the source files in this directory.
set(sky_SRC
    src/oskar_angles_from_lm.c
    src/oskar_rotate_sources.cpp

    src/oskar_filter_sources_by_flux.cpp
    src/oskar_filter_sources_by_radius.cpp
    src/oskar_generate_random_sources.cpp
    src/oskar_cuda_compute_local_sky.c
    src/oskar_cuda_horizon_clip.cu
    src/oskar_cuda_ra_dec_to_az_el.cu
    src/oskar_cuda_ra_dec_to_hor_lmn.cu
    src/oskar_cuda_ra_dec_to_relative_lmn.cu
    src/oskar_cuda_transform_to_local_stokes.cu
    src/oskar_date_time_to_mjd.c
    src/oskar_equation_of_equinoxes_fast.c
    src/oskar_evaluate_jones_R.cu
    src/oskar_ha_dec_to_hor_lmn.c
    src/oskar_mjd_to_gmst.c
    src/oskar_mjd_to_last_fast.c
    src/oskar_mjd_to_lmst.c
    src/oskar_ra_dec_to_hor_lmn.c

    src/oskar_sky_model_append.cpp
    src/oskar_sky_model_load.cpp
    src/oskar_sky_model_resize.cpp
    src/oskar_sky_model_set_source.cpp
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
