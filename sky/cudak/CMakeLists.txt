#
# src/sky/cudak/CMakeLists.txt
#

set(sky_cuda_kernels_SRC
    src/oskar_cudak_ha_dec_to_az_el.cu
    src/oskar_cudak_ha_dec_to_hor_lmn.cu
    src/oskar_cudak_hor_lmn_to_az_el.cu
    src/oskar_cudak_hor_lmn_to_phi_theta.cu
    src/oskar_cudak_lm_to_n.cu
    src/oskar_cudak_ra_dec_to_az_el.cu
    src/oskar_cudak_ra_dec_to_hor_lmn.cu
    src/oskar_cudak_scale_brightness_by_spectral_index.cu
    src/oskar_cudak_update_horizon_mask.cu
)
set(sky_cuda_kernels_SRC "${sky_cuda_kernels_SRC}" PARENT_SCOPE)
