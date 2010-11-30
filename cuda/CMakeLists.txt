#
# src/cuda
#

set(cuda_SRC
    # CUDA kernels:
    src/_antennaSignals.cu
    src/_beamPatternDirect.cu
    src/_beamPatternWeights.cu
    src/_generateWeights.cu
    # Callable functions:
    src/antennaSignals.cu
    src/beamPatternDirect.cu
    src/beamPatternWeights.cu
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
