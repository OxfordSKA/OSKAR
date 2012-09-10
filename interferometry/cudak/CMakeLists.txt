#
# src/interferometry/cudak/CMakeLists.txt
#

set(interferometry_cuda_kernels_SRC
    src/oskar_cudak_correlator_extended.cu
    src/oskar_cudak_correlator.cu
    src/oskar_cudak_correlator_scalar.cu
)
set(interferometry_cuda_kernels_SRC "${interferometry_cuda_kernels_SRC}" PARENT_SCOPE)
