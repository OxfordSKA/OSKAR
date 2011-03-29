#
# src/cuda/kernels/CMakeLists.txt
#
#==============================================================================

set(cuda_kernels_SRC
    src/oskar_cudak_as2hi.cu
    src/oskar_cudak_bp2hiw.cu
    src/oskar_cudak_bp2hiws.cu
    src/oskar_cudak_bp2hugw.cu
    src/oskar_cudak_bp2hcgw.cu
    src/oskar_cudak_eq2hg.cu
    src/oskar_cudak_im2dft.cu
    src/oskar_cudak_le2hg.cu
    src/oskar_cudak_pc2ht.cu
    src/oskar_cudak_rpw3leg.cu
    src/oskar_cudak_vecadd.cu
    src/oskar_cudak_wt2hg.cu
    src/oskar_cudak_wt2hgu.cu
    src/oskar_cudakd_bp2hiw.cu
    src/oskar_cudakd_bp2hiws.cu
    src/oskar_cudakd_bp2hugw.cu
    src/oskar_cudakd_bp2hcgw.cu
    src/oskar_cudakd_eq2hg.cu
    src/oskar_cudakd_im2dft.cu
    src/oskar_cudakd_le2hg.cu
    src/oskar_cudakd_pc2ht.cu
    src/oskar_cudakd_rpw3leg.cu
    src/oskar_cudakd_wt2hg.cu
    src/oskar_cudakd_wt2hgu.cu
)
set(cuda_kernels_SRC "${cuda_kernels_SRC}" PARENT_SCOPE)

