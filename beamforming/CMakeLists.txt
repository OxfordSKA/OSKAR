#
# src/beamforming/CMakeLists.txt
#
#==============================================================================

set(beamforming_SRC
    src/beamPattern.cpp
)
set(beamforming_SRC "${beamforming_SRC}" PARENT_SCOPE)

add_library(oskar_beamforming ${beamforming_SRC})

set_target_properties(oskar_beamforming PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)

# === Recurse into test subdirectory.
add_subdirectory(test)
