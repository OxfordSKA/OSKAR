#
# src/station/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)

    set(station_test_SRC
        src/CppUnitMain.cpp
        #src/Test_dipole_pattern.cu
        src/Test_dipole_orientation.cpp
        src/Test_element_weights_errors.cpp
        src/Test_evaluate_array_pattern.cpp
        src/Test_evaluate_pierce_points.cpp
        src/Test_evaluate_station_beam_dipoles.cpp
        src/Test_evaluate_station_beam.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
    cuda_add_executable(station_test ${station_test_SRC})
    set_target_properties(station_test PROPERTIES
       COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
       LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    target_link_libraries(station_test
        oskar
        ${CPPUNIT_LIBRARIES}
        ${CUDA_LIBRARIES})

    add_test(station_test station_test)

endif()


set(name station_test2)
include_directories(${PROJECT_SOURCE_DIR}/extern/gtest-1.6.0/include)
set(${name}_SRC
    src/Test_evaluate_jones_E.cpp
    src/Test_StationModel.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest_main)
set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
add_test(station_test ${name})


# Array pattern benchmark binary.
include_directories(${CUDA_INCLUDE_DIRS})
set(name oskar_array_pattern_benchmark)
add_executable(${name} src/${name}.cpp)
target_link_libraries(${name} oskar ${CUDA_LIBRARIES})

