#
# src/interferometry/test/CMakeLists.txt
#

set(name interferometry_test)
set(${name}_SRC
    main.cpp
    Test_add_system_noise.cpp
    Test_correlate.cpp
    Test_evaluate_baselines.cpp
    Test_station_coord_transforms.cpp
    Test_TelescopeModel.cpp
    Test_Visibilities.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)
set_target_properties(${name} PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")
add_test(interferometry_test ${name})


# Correlate kernel performance test.
set(name correlate_performance_test)
cuda_add_executable(${name} Test_correlate_performance.cu)
target_link_libraries(${name} oskar gtest_main ${CUDA_LIBRARIES})

# Correlator benchmark binary.
include_directories(${CUDA_INCLUDE_DIRS})
set(name oskar_correlator_benchmark)
add_executable(${name} ${name}.cpp)
target_link_libraries(${name} oskar ${CUDA_LIBRARIES})
