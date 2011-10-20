#
# src/station/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)

    include_directories(${CPPUNIT_INCLUDE_DIR})
    #include_directories(${QT_INCLUDE_DIR})


    set(station_test_SRC
        src/CppUnitMain.cu
        #src/BeamPatternTest.cpp
        src/ElementModelTest.cpp
    )

    cuda_add_executable(stationTest ${station_test_SRC})

    set_target_properties(stationTest PROPERTIES
       COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
       LINK_FLAGS "${OpenMP_CXX_FLAGS}")

    target_link_libraries(stationTest oskar
        ${CPPUNIT_LIBRARIES}
        #${QT_QTCORE_LIBRARIES} #FIXME dependency
    )

    add_test(stationTest stationTest)

endif()

