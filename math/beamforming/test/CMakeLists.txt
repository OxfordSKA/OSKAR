#
# src/math/beamforming/test
#



# CppUnit
# ==============================================================================
set(math_beamforming_test_SRC
    src/CppUnitMain.cpp
    src/BeamPatternTest.cpp
)

include_directories()

add_executable(beamformingTest ${math_beamforming_test_SRC})

set_target_properties(beamformingTest PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
)

target_link_libraries(beamformingTest
    math_beamforming
    ${CPPUNIT_LIBRARIES}
)
