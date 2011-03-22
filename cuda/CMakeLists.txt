#
# src/cuda/CMakeLists.txt
#
#==============================================================================

set(cuda_SRC
    # CUDA kernels:
    kernels/src/oskar_cudak_as2hi.cu
    kernels/src/oskar_cudak_bp2hiw.cu
    kernels/src/oskar_cudak_bp2hiws.cu
    kernels/src/oskar_cudak_bp2hugw.cu
    kernels/src/oskar_cudak_bp2hcgw.cu
    kernels/src/oskar_cudak_eq2hg.cu
    kernels/src/oskar_cudak_im2dft.cu
    kernels/src/oskar_cudak_le2hg.cu
    kernels/src/oskar_cudak_pc2ht.cu
    kernels/src/oskar_cudak_rpw3leg.cu
    kernels/src/oskar_cudak_vecadd.cu
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
    src/oskar_cuda_hbp2hig.cu
    src/oskar_cuda_im2dft.cu
    src/oskar_cuda_im2dftlm.cu
    src/oskar_cuda_le2hg.cu
    src/oskar_cuda_rpw3leg.cu
    src/oskar_cuda_vecadd.cu
)

# === Macros to build the oskar-lib cuda module.
include_directories(${CUDA_INCLUDE_DIRS})
cuda_add_library(oskar_cuda ${cuda_SRC})
target_link_libraries(oskar_cuda ${CUDA_CUBLAS_LIBRARIES} ${CUDA_LIBRARIES})


# === Install configuration for the oskar-lib cuda module.
install(TARGETS oskar_cuda LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR})
set_target_properties(oskar_cuda PROPERTIES
    SOVERSION ${oskar-lib_SOVERSION})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.h"
    PATTERN "src" EXCLUDE
    PATTERN "kernels" EXCLUDE
    PATTERN "test" EXCLUDE)


# === Add a target for clean of nvcc generated files.
cuda_build_clean_target()

# === Recurse into test subdirectory.
add_subdirectory(test)
