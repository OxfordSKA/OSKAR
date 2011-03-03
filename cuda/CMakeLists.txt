#
# src/cuda
#

set(cuda_SRC
    # CUDA kernels:
    kernels/src/oskar_cudak_as2hi.cu
    kernels/src/oskar_cudak_bp2hiw.cu
    kernels/src/oskar_cudak_bp2hugw.cu
    kernels/src/oskar_cudak_bp2hcgw.cu
    kernels/src/oskar_cudak_eq2hg.cu
    kernels/src/oskar_cudak_im2dft.cu
    kernels/src/oskar_cudak_le2hg.cu
    kernels/src/oskar_cudak_pc2ht.cu
    kernels/src/oskar_cudak_rpw3leg.cu
    kernels/src/oskar_cudak_wt2hg.cu
    kernels/src/oskar_cudak_wt2hgu.cu
    # Callable functions:
    src/oskar_cuda_as2hi.cu
    src/oskar_cuda_bf2hig.cu
    src/oskar_cuda_bfmv.cu
    src/oskar_cuda_bp2hcgg.cu
    src/oskar_cuda_bp2hcggu.cu
    src/oskar_cuda_bp2hig.cu
    src/oskar_cuda_bp2higu.cu
    src/oskar_cuda_bp2hugg.cu
    src/oskar_cuda_bp2huggu.cu
    src/oskar_cuda_eq2hg.cu
    src/oskar_cuda_im2dft.cu
    src/oskar_cuda_im2dftlm.cu
    src/oskar_cuda_le2hg.cu
    src/oskar_cuda_rpw3leg.cu
    src/vectorAdd.cu
)

CUDA_ADD_library(oskar_cuda ${cuda_SRC} SHARED)

include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(oskar_cuda
    ${CUDA_CUBLAS_LIBRARIES}
    ${CUDA_LIBRARIES}
)


# === Adds a special command to clean nvcc generated files.
cuda_build_clean_target()

# === Recurse into test subdirectory.
add_subdirectory(test)
