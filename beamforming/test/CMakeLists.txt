#
# src/beamforming/test
#



# CppUnit
# ==============================================================================
if (CPPUNIT_FOUND)

    set(beamforming_test_SRC
        src/CppUnitMain.cpp
        src/BeamPatternTest.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR})

    add_executable(beamformingTest ${beamforming_test_SRC})

    set_target_properties(beamformingTest PROPERTIES
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        LINK_FLAGS "${OpenMP_CXX_FLAGS}")

    target_link_libraries(beamformingTest
        oskar_beamforming
        ${CPPUNIT_LIBRARIES})

endif()
