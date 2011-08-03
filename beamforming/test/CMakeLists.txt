#
# src/beamforming/test
#



# CppUnit
# ==============================================================================
if (CPPUNIT_FOUND)
    include_directories(${CPPUNIT_INCLUDE_DIR})

    set(beamforming_test_SRC
        src/CppUnitMain.cpp
        src/BeamPatternTest.cpp
    )

    add_executable(beamformingTest ${beamforming_test_SRC})
    set_target_properties(beamformingTest PROPERTIES
       COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    target_link_libraries(beamformingTest oskar ${CPPUNIT_LIBRARIES})

endif()

