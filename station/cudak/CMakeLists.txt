#
# src/station/cudak/CMakeLists.txt
#

set(station_cuda_kernels_SRC
    src/oskar_cudak_antenna.cu
    src/oskar_cudak_apodisation.cu
    src/oskar_cudak_blank_below_horizon.cu
)
set(station_cuda_kernels_SRC "${station_cuda_kernels_SRC}" PARENT_SCOPE)

