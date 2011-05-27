#
# src/cuda/kernels/CMakeLists.txt
#
#==============================================================================

set(cuda_kernels_SRC
    src/oskar_cudak_antenna.cu
    src/oskar_cudak_apodisation.cu
    src/oskar_cudak_as2hi.cu
    src/oskar_cudak_bp2hiw.cu
    src/oskar_cudak_bp2phiw.cu
    src/oskar_cudak_bp2hiws.cu
    src/oskar_cudak_bp2hugw.cu
    src/oskar_cudak_cmatmul.cu
    src/oskar_cudak_correlator.cu
    src/oskar_cudak_crvecmul.cu
    src/oskar_cudak_cvecmul.cu
    src/oskar_cudak_cvecset.cu
    src/oskar_cudak_dft_c2r_2d.cu
    src/oskar_cudak_dftsw_c2c_2d.cu
    src/oskar_cudak_dftw_o2c_2d.cu
    src/oskar_cudak_eq2hg.cu
    src/oskar_cudak_im2dft.cu
    src/oskar_cudak_le2hg.cu
    src/oskar_cudak_pc2ht.cu
    src/oskar_cudak_rpw3leg.cu
    src/oskar_cudak_rpw3leglm.cu
    src/oskar_cudak_vecadd.cu
    src/oskar_cudak_vecmul.cu
    src/oskar_cudak_wt2hg.cu
    src/oskar_cudak_wt2phg.cu
)
set(cuda_kernels_SRC "${cuda_kernels_SRC}" PARENT_SCOPE)

