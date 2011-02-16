#
# src/cuda
#

set(cuda_SRC
    # CUDA kernels:
    src/_antennaSignal2dHorizontalIsotropic.cu
    src/_beamPattern2dHorizontalWeights.cu
    src/_equatorialToHorizontalGeneric.cu
    src/_imager2dDft.cu
    src/_precompute2dHorizontalTrig.cu
    src/_roundRobin.cu
    src/_weights2dHorizontalGeometric.cu
    # Callable functions:
    src/antennaSignal2dHorizontalIsotropic.cu
    src/beamformer2dHorizontalIsotropicGeometric.cu
    src/beamformerMatrixVector.cu
    src/beamPattern2dHorizontalGeometric.cu
    src/imager2dDft.cu
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
