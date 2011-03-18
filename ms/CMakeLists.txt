#
# src/ms/CMakeLists.txt
#

set(ms_SRC
    #src/oskar_ms_create_uvw1.cu
    src/MsCreate.cpp
)

include_directories(${CASACORE_INCLUDE_DIR}/casacore)
add_library(oskar_ms ${ms_SRC})
target_link_libraries(oskar_ms ${CASACORE_LIBRARIES})

# === Recurse into test subdirectory.
add_subdirectory(test)
