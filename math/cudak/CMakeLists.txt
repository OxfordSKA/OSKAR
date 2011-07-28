#
# src/math/cudak/CMakeLists.txt
#

set(math_cuda_kernels_SRC
    src/oskar_math_cudak_dft_c2r_2d.cu
    src/oskar_math_cudak_dftsw_c2c_2d.cu
    src/oskar_math_cudak_dftw_2d_seq_in.cu
    src/oskar_math_cudak_dftw_3d_seq_out.cu
    src/oskar_math_cudak_dftw_o2c_2d.cu
    src/oskar_math_cudak_mat_mul_cc.cu
    src/oskar_math_cudak_sph_to_lm.cu
    src/oskar_math_cudak_vec_add_rr.cu
    src/oskar_math_cudak_vec_mul_cc.cu
    src/oskar_math_cudak_vec_mul_cr.cu
    src/oskar_math_cudak_vec_mul_rr.cu
    src/oskar_math_cudak_vec_set_c.cu
    src/oskar_math_cudak_vec_sub_rr.cu
    src/oskar_math_cudak_vec_sub_sr.cu
)
set(math_cuda_kernels_SRC "${math_cuda_kernels_SRC}" PARENT_SCOPE)

