#
# src/math/cudak/CMakeLists.txt
#

set(math_cuda_kernels_SRC
    src/oskar_cudak_sph_to_lm.cu
    src/oskar_cudak_vec_sub_sr.cu
)
set(math_cuda_kernels_SRC "${math_cuda_kernels_SRC}" PARENT_SCOPE)

