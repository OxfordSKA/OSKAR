#
# src/math/CMakeLists.txt
#

add_subdirectory(cudak)

# Add the source files in this directory.
set(math_SRC
    src/oskar_angular_distance.c
    src/oskar_cart2sph.c
    src/oskar_dft_c2r_2d_cuda.cu
    src/oskar_dftw_c2c_2d_cuda.cu
    src/oskar_dftw_c2c_3d_cuda.cu
    src/oskar_dftw_m2m_2d_cuda.cu
    src/oskar_dftw_m2m_3d_cuda.cu
    src/oskar_dftw_o2c_2d_cuda.cu
    src/oskar_dftw_o2c_3d_cuda.cu
    src/oskar_dierckx_bispev_bicubic_cuda.cu
    src/oskar_dierckx_bispev_cuda.cu
    src/oskar_dierckx_bispev.c
    src/oskar_dierckx_fpback.c
    src/oskar_dierckx_fpbisp.c
    src/oskar_dierckx_fpbspl.c
    src/oskar_dierckx_fpdisc.c
    src/oskar_dierckx_fpgivs.c
    src/oskar_dierckx_fporde.c
    src/oskar_dierckx_fprank.c
    src/oskar_dierckx_fprati.c
    src/oskar_dierckx_fprota.c
    src/oskar_dierckx_fpsurf.c
    src/oskar_dierckx_surfit.c
    src/oskar_find_closest_match.c
    src/oskar_fit_ellipse.c
    src/oskar_gaussian.c
    src/oskar_gaussian_cuda.cu
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
    src/oskar_matrix_multiply.c
    src/oskar_mean.c
    src/oskar_meshgrid.c
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
    src/oskar_spline_data_copy.c
    src/oskar_spline_data_evaluate.c
    src/oskar_spline_data_free.c
    src/oskar_spline_data_init.c
    src/oskar_spline_data_location.c
    src/oskar_spline_data_surfit.c
    src/oskar_spline_data_type.c
)
foreach (file ${math_cuda_kernels_SRC})
    list(APPEND math_SRC cudak/${file})
endforeach ()

set(math_SRC "${math_SRC}" PARENT_SCOPE)

add_subdirectory(test)
