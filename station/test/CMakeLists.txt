#
# src/station/test
#


if (CPPUNIT_FOUND)

    include_directories(${CPPUNIT_INCLUDE_DIR})

    set(station_test_SRC
        src/CppUnitMain.cpp
        src/BeamPatternTest.cpp
    )

    add_executable(stationTest ${station_test_SRC})
    set_target_properties(stationTest PROPERTIES
       COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    target_link_libraries(stationTest oskar ${CPPUNIT_LIBRARIES})

endif()

