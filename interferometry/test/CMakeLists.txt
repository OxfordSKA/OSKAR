#
# src/interferometry/test/CMakeLists.txt
#

if (CPPUNIT_FOUND)
    set(interferometry_test_SRC
        src/CppUnitMain.cpp
        src/Test_add_system_noise.cpp
    )

    include_directories(${CPPUNIT_INCLUDE_DIR} ${CUDA_INCLUDE_DIRS})
    cuda_add_executable(interferometry_test ${interferometry_test_SRC})
    target_link_libraries(interferometry_test oskar
        ${CPPUNIT_LIBRARIES} ${CUDA_LIBRARIES}
    )
    add_test(interferometry_test interferometry_test)
endif()


# New test using gtest.
set(name interferometry_test2)
include_directories(${PROJECT_SOURCE_DIR}/extern/gtest-1.6.0/include)
set(${name}_SRC
    src/Test_correlate.cpp
    src/Test_evaluate_baselines.cpp
    src/Test_station_coord_transforms.cpp
    src/Test_TelescopeModel.cpp
    src/Test_Visibilities.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest_main)
set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
add_test(interferometry_test ${name})


# Correlate kernel performance test.
set(name correlate_performance_test)
include_directories(${PROJECT_SOURCE_DIR}/extern/gtest-1.6.0/include)
cuda_add_executable(${name} src/Test_correlate_performance.cu)
target_link_libraries(${name} oskar gtest_main ${CUDA_LIBRARIES})


if (NOT WIN32)
    # Correlator benchmark binary.
    include_directories(${CUDA_INCLUDE_DIRS})
    set(name oskar_correlator_benchmark)
    add_executable(${name} src/${name}.cpp)
    target_link_libraries(${name} oskar ${CUDA_LIBRARIES})
endif (NOT WIN32)
