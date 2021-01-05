#
# oskar/math/CMakeLists.txt
#

set(math_SRC
    define_dft_c2r.h
    define_dftw_c2c.h
    define_dftw_m2m.h
    define_fftphase.h
    define_gaussian_circular.h
    define_legendre_polynomial.h
    define_multiply.h
    define_prefix_sum.h
    define_spherical_harmonic.h
    src/oskar_angular_distance.c
    src/oskar_bearing_angle.c
    src/oskar_dft_c2r.c
    src/oskar_dftw.c
    src/oskar_ellipse_radius.c
    src/oskar_evaluate_image_lon_lat_grid.c
    src/oskar_evaluate_image_lm_grid.c
    src/oskar_evaluate_image_lmn_grid.c
    src/oskar_fft.c
    src/oskar_fftpack_cfft.c
    src/oskar_fftpack_cfft_f.c
    src/oskar_fftphase.c
    src/oskar_find_closest_match.c
    src/oskar_fit_ellipse.c
    src/oskar_gaussian_circular.c
    src/oskar_healpix_npix_to_nside.c
    src/oskar_lapack_subset.c
    src/oskar_linspace.c
    src/oskar_math_cpu.cl
    src/oskar_math_gpu.cl
    src/oskar_math.cl
    src/oskar_matrix_multiply.c
    src/oskar_meshgrid.c
    src/oskar_prefix_sum.c
    src/oskar_prefix_sum_cpu.cl
    src/oskar_prefix_sum_gpu.cl
    src/oskar_random_broken_power_law.c
    src/oskar_random_gaussian.c
    src/oskar_random_power_law.c
    src/oskar_rotate.c
    src/oskar_round_robin.c
    #src/oskar_spherical_harmonic_sum.c
    #src/oskar_spherical_harmonic.c
    #src/oskar_sph_rotate_to_position.c
)

if (CUDA_FOUND)
    list(APPEND math_SRC src/oskar_math.cu)
endif()

set(math_SRC "${math_SRC}" PARENT_SCOPE)

if (BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
    add_subdirectory(test)
endif()
