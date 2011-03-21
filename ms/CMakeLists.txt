#
# src/ms/CMakeLists.txt
#

set(ms_SRC
    src/MsAppend.cpp
    src/MsCreate.cpp
    src/MsManip.cpp
    src/oskar_ms_append_vis1.cpp
    src/oskar_ms_create_empty.cpp
    src/oskar_ms_create_meta1.cpp
    src/oskar_ms_create_vis1.cpp
)

include_directories(${CASACORE_INCLUDE_DIR}/casacore)
add_library(oskar_ms ${ms_SRC})
set_target_properties(oskar_ms PROPERTIES COMPILE_FLAGS "-Wno-cast-qual")
target_link_libraries(oskar_ms ${CASACORE_LIBRARIES})


# === Recurse into test subdirectory.
add_subdirectory(test)
