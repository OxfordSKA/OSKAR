#
# src/interferometry/cudak/CMakeLists.txt
#

set(interferometry_cuda_kernels_SRC
    src/oskar_cudak_correlator_extended.cu
    src/oskar_cudak_correlator_scalar.cu
    src/oskar_cudak_correlator_time_smearing_extended.cu
    src/oskar_cudak_correlator_time_smearing.cu
    src/oskar_cudak_correlator.cu
)
set(interferometry_cuda_kernels_SRC "${interferometry_cuda_kernels_SRC}" PARENT_SCOPE)
