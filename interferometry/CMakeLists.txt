#
# src/interferometry/CMakeLists.txt
#


# ==== Add the source files in this directory.
set(interferometry_SRC
    # CUDA kernels
    #cudak/src/oskar_cudak_correlator.cu
    cudak/src/oskar_cudak_xyz2uvw.cu

    # C/C++ interface
    src/oskar_compute_baselines.c
    src/oskar_cuda_correlator_lm_bw.cu
    src/oskar_cuda_correlator_scalar.cu
    #src/oskar_cuda_interferometer1.cu
    src/oskar_xyz2uvw.c
)
set(interferometry_SRC "${interferometry_SRC}" PARENT_SCOPE)

add_subdirectory(test)
add_subdirectory(matlab)
