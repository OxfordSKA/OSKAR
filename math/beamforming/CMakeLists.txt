#
# src/math/beamforming
#

set(math_beamforming_SRC
    src/beamPattern.cpp
)

include_directories(
    ${QT_QTCORE_INCLUDE_DIR}
)

add_library(math_beamforming ${math_beamforming_SRC})

set_target_properties(math_beamforming PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)

target_link_libraries(math_beamforming
    ${QT_QTCORE_LIBRARY}
)

# Recurse into test subdirectory.
add_subdirectory(test)
