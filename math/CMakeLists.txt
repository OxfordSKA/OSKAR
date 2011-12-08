#
# src/math/CMakeLists.txt
#

include_directories(${oskar-lib_SOURCE_DIR}/math)

add_subdirectory(matlab)
add_subdirectory(cudak)

# Add the source files in this directory.
set(math_SRC
    src/oskar_cuda_dft_c2r_2d.cu
    src/oskar_cuda_interp_bilinear.cu
    src/oskar_cuda_jones_mul_c2.cu
    src/oskar_cuda_jones_mul_mat1_c1.cu
    src/oskar_cuda_jones_mul_mat2.cu
    src/oskar_cuda_jones_mul_scalar_c2.cu
    src/oskar_cuda_jones_set_real_scalar_1.cu
    src/oskar_cuda_jones_set_real_scalar_4.cu
    src/oskar_jones_copy.c
    src/oskar_jones_free.c
    src/oskar_jones_get_station_pointer.c
    src/oskar_jones_init.c
    src/oskar_jones_join.cpp
    src/oskar_jones_set_size.c
    src/oskar_jones_set_real_scalar.c
    src/oskar_Jones.cpp
    src/oskar_linspace.c
    src/oskar_mat_tri_c.c
    src/oskar_meshgrid.c
    src/oskar_round_robin.c
    src/oskar_sph_from_lm.c
    src/oskar_sph_to_lm.c
    src/oskar_spline_data_compute.c
    src/oskar_spline_data_evaluate.c
    src/oskar_spline_data_free.c
    src/oskar_spline_data_init.c
    src/oskar_spline_surface_evaluate.c
    src/oskar_spline_surface_init.c
)
foreach (file ${math_cuda_kernels_SRC})
    list(APPEND math_SRC cudak/${file})
endforeach ()

set(math_SRC "${math_SRC}" PARENT_SCOPE)

add_subdirectory(test)
