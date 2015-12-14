#
# src/math/CMakeLists.txt
#

# Add the source files in this directory.
set(math_SRC
    src/oskar_angular_distance.c
    src/oskar_bearing_angle.c
    src/oskar_dft_c2r_2d_omp.c
    src/oskar_dftw_c2c_2d_omp.c
    src/oskar_dftw_c2c_3d_omp.c
    src/oskar_dftw_m2m_2d_omp.c
    src/oskar_dftw_m2m_3d_omp.c
    src/oskar_dftw_o2c_2d_omp.c
    src/oskar_dftw_o2c_3d_omp.c
    src/oskar_ellipse_radius.c
    src/oskar_find_closest_match.c
    src/oskar_fit_ellipse.c
    src/oskar_gaussian.c
    src/oskar_healpix_npix_to_nside.c
    src/oskar_linspace.c
    src/oskar_matrix_multiply.c
    src/oskar_meshgrid.c
    src/oskar_random_broken_power_law.c
    src/oskar_random_gaussian.c
    src/oskar_random_power_law.c
    src/oskar_rotate.c
    src/oskar_round_robin.c
    #src/oskar_sph_rotate_to_position.c
)

if (CUDA_FOUND)
    list(APPEND math_SRC
        src/oskar_dft_c2r_2d_cuda.cu
        src/oskar_dftw_c2c_2d_cuda.cu
        src/oskar_dftw_c2c_3d_cuda.cu
        src/oskar_dftw_m2m_2d_cuda.cu
        src/oskar_dftw_m2m_3d_cuda.cu
        #src/oskar_dftw_m2m_3d_indexed_input_cuda.cu
        src/oskar_dftw_o2c_2d_cuda.cu
        src/oskar_dftw_o2c_3d_cuda.cu
        src/oskar_gaussian_cuda.cu
    )
endif()

set(math_SRC "${math_SRC}" PARENT_SCOPE)

add_subdirectory(test)
