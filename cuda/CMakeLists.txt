#
# src/cuda
#

set(cuda_SRC
    # CUDA kernels:
    src/_antennaSignal2dHorizontalIsotropic.cu
    src/_beamPattern2dHorizontalGeometric.cu
    src/_beamPattern2dHorizontalWeights.cu
    src/_equatorialToHorizontalGeneric.cu
    src/_precompute2dHorizontalTrig.cu
    src/_weights2dHorizontalGeometric.cu
    # Callable functions:
    src/antennaSignal2dHorizontalIsotropic.cu
    src/beamPattern2dHorizontalGeometric.cu
    src/beamPattern2dHorizontalWeights.cu
    src/vectorAdd.cu
)

CUDA_ADD_library(cuda ${cuda_SRC})

include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(cuda
    ${CUDA_LIBRARIES}
    ${QT_QTCORE_LIBRARY}
)


# === Adds a special command to clean nvcc generated files.
cuda_build_clean_target()

# === Recurse into test subdirectory.
add_subdirectory(test)
