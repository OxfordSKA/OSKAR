#
# src/math/CMakeLists.txt
#

include_directories(${oskar-lib_SOURCE_DIR}/math)

add_subdirectory(matlab)
add_subdirectory(cudak)

# Add the source files in this directory.
set(math_SRC
    src/oskar_angular_distance.c
    src/oskar_cart2sph.c
    src/oskar_cuda_dft_c2r_2d.cu
    src/oskar_fit_ellipse.c
    src/oskar_healpix_angles_to_pix_ring.c
    src/oskar_healpix_npix_to_nside.c
    src/oskar_healpix_nside_to_npix.c
    src/oskar_healpix_pix_to_angles_ring.c
    src/oskar_jones_copy.c
    src/oskar_jones_free.c
    src/oskar_jones_get_station_pointer.c
    src/oskar_jones_init.c
    src/oskar_jones_join.c
    src/oskar_jones_set_size.c
    src/oskar_jones_set_real_scalar.c
    src/oskar_Jones.cpp
    src/oskar_linspace.c
    src/oskar_matrix_invert.c
    src/oskar_matrix_multiply.c
    src/oskar_mean.c
    src/oskar_meshgrid.c
    #src/oskar_mrdivide.c ## Do not use.. doesnt currently catch all cases!
    src/oskar_random_broken_power_law.c
    src/oskar_random_gaussian.c
    src/oskar_random_power_law.c
    src/oskar_random_uniform.c
    src/oskar_rotate.c
    src/oskar_round_robin.c
    src/oskar_sph_from_lm.c
    src/oskar_sph_rotate_points.c
    src/oskar_sph_rotate_to_position.c
    src/oskar_sph_to_lm.c
    src/oskar_sph2cart.c
    src/oskar_spline_data_compute_regrid.c # Deprecated
    src/oskar_spline_data_compute_sphere.c
    src/oskar_spline_data_compute_surfit.c
    src/oskar_spline_data_evaluate.c
    src/oskar_spline_data_free.c
    src/oskar_spline_data_init.c
    src/oskar_spline_data_location.c
    src/oskar_spline_data_type.c
)
foreach (file ${math_cuda_kernels_SRC})
    list(APPEND math_SRC cudak/${file})
endforeach ()

set(math_SRC "${math_SRC}" PARENT_SCOPE)

add_subdirectory(test)
