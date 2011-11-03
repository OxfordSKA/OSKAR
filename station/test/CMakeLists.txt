#
# src/station/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)

    include_directories(${CPPUNIT_INCLUDE_DIR})

    set(station_test_SRC
        src/CppUnitMain.cu
        #src/BeamPatternTest.cpp
        src/ElementModelTest.cpp
        #src/evaluate_jones_E_test.cpp
        src/evaluate_station_beam_test.cpp
        src/StationModelTest.cpp
    )

    cuda_add_executable(stationTest ${station_test_SRC})

    set_target_properties(stationTest PROPERTIES
       COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
       LINK_FLAGS "${OpenMP_CXX_FLAGS}")

    target_link_libraries(stationTest oskar ${CPPUNIT_LIBRARIES})

    add_test(stationTest stationTest)

endif()

