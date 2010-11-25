#
# src/cuda
#

set(cuda_SRC
    src/vectorAdd.cu
    src/beamPattern.cu
)

CUDA_ADD_library(cuda ${cuda_SRC})

include_directories(${CUDA_INCLUDE_DIRS})

target_link_libraries(cuda
    ${CUDA_LIBRARIES}
)


# === Adds a special command to clean nvcc generated files.
cuda_build_clean_target()

# === Recurse into test subdirectory.
add_subdirectory(test)
