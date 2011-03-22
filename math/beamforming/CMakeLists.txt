#
# src/math/beamforming/CMakeLists.txt
#
#==============================================================================

# === Source files for the math beamforming sub-module.
set(math_beamforming_SRC
    src/beamPattern.cpp
)
set(math_beamforming_SRC "${math_beamforming_SRC}" PARENT_SCOPE)

# === Build a beamforming sub-module for testing (this is NOT installed).
include_directories(${QT_QTCORE_INCLUDE_DIR})
add_library(oskar_math_beamforming ${math_beamforming_SRC})
set_target_properties(oskar_math_beamforming PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)
target_link_libraries(oskar_math_beamforming
    ${QT_QTCORE_LIBRARY}
)

# === Recurse into test subdirectory.
add_subdirectory(test)
