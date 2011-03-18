#
# src/ms
#

set(ms_SRC
    #src/oskar_ms_create_uvw1.cu
    src/MsCreate.cpp
)

add_library(oskar_ms ${ms_SRC} SHARED)

include_directories(${CUDA_INCLUDE_DIRS})


# === Recurse into test subdirectory.
add_subdirectory(test)
