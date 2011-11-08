#
# src/cuda/kernels/CMakeLists.txt
#
#==============================================================================

set(cuda_kernels_SRC
    src/oskar_cudak_as2hi.cu
    src/oskar_cudak_bp2hiw.cu
    src/oskar_cudak_bp2phiw.cu
    src/oskar_cudak_bp2hiws.cu
    src/oskar_cudak_bp2hugw.cu
    #src/oskar_cudak_eq2hg.cu
    src/oskar_cudak_im2dft.cu
    #src/oskar_cudak_le2hg.cu
    src/oskar_cudak_pc2ht.cu
    src/oskar_cudak_wt2hg.cu
    src/oskar_cudak_wt2phg.cu
)
set(cuda_kernels_SRC "${cuda_kernels_SRC}" PARENT_SCOPE)

