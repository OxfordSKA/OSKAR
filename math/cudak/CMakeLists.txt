#
# src/math/cudak/CMakeLists.txt
#

set(math_cuda_kernels_SRC
    src/oskar_cudak_dft_c2r_2d.cu
    src/oskar_cudak_dftsw_c2c_2d.cu
    src/oskar_cudak_dftw_2d.cu
    src/oskar_cudak_dftw_3d.cu
    src/oskar_cudak_dftw_o2c_2d.cu
    src/oskar_cudak_dftw_o2c_3d.cu
    src/oskar_cudak_gaussian.cu
    src/oskar_cudak_sph_to_lm.cu
    src/oskar_cudak_vec_sub_sr.cu
)
set(math_cuda_kernels_SRC "${math_cuda_kernels_SRC}" PARENT_SCOPE)

