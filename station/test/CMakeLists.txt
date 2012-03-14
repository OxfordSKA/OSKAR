#
# src/station/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)

    set(station_test_SRC
        src/CppUnitMain.cpp
        #src/BeamPatternTest.cpp
        #src/Test_ElementModel.cpp
        src/Test_dipole_pattern.cu
        src/Test_element_weights_errors.cpp
        src/Test_evaluate_jones_E.cpp
        src/Test_evaluate_station_beam_dipoles.cu
        src/Test_evaluate_station_beam.cpp
        src/Test_StationModel.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
    cuda_add_executable(station_test ${station_test_SRC})
    set_target_properties(station_test PROPERTIES
       COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
       LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    target_link_libraries(station_test oskar ${CPPUNIT_LIBRARIES} ${CUDA_LIBRARIES})

    add_test(station_test station_test)

endif()

# Add standalone test for Meerkat beam.
add_executable(meerkat_beam_test src/meerkat_beam_test.cpp)
target_link_libraries(meerkat_beam_test oskar ${CUDA_LIBRARIES})
    set_target_properties(meerkat_beam_test PROPERTIES
       COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
       LINK_FLAGS "${OpenMP_CXX_FLAGS}")
