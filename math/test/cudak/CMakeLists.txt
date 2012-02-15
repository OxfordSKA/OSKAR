#
# src/math/test/cudak/CMakeLists.txt
#

set(math_test_cuda_kernels_SRC
    src/test_curand_generate.cu 
)
cuda_add_library(oskar_math_test_cudak ${math_test_cuda_kernels_SRC})
