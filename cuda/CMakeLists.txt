#
# src/cuda/CMakeLists.txt
#

# ==== Recurse into the kernels sub-directory.
add_subdirectory(kernels)

# ==== Create list of source files for the oskar-lib cuda module.
set(cuda_SRC
    src/oskar_cuda_as2hi.cu
    src/oskar_cuda_bf2hig.cu

    src/oskar_cuda_bp2hc.cu
    src/oskar_cuda_bp2hugg.cu
    src/oskar_cuda_bp2hwcr.cu
    src/oskar_cuda_bp2phc.cu

    src/oskar_cuda_dft_o2c_2d.cu

    src/oskar_cuda_eq2hg.cu
    src/oskar_cuda_hbp2hig.cu
    src/oskar_cuda_im2dftlm.cu
    src/oskar_cuda_le2hg.cu
)
foreach (file ${cuda_kernels_SRC})
    list(APPEND cuda_SRC kernels/${file})
endforeach ()
set(cuda_SRC "${cuda_SRC}" PARENT_SCOPE)

# ==== Recurse into test subdirectory.
add_subdirectory(test)
