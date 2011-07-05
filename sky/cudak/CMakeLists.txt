#
# src/sky/cudak/CMakeLists.txt
#
#==============================================================================

set(sky_cuda_kernels_SRC
    src/oskar_sky_cudak_ha_dec_to_az_el.cu
    src/oskar_sky_cudak_ha_dec_to_hor_lmn.cu
    src/oskar_sky_cudak_ra_dec_to_az_el.cu
)
set(sky_cuda_kernels_SRC "${sky_cuda_kernels_SRC}" PARENT_SCOPE)
