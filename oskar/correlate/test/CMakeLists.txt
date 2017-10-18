#
# oskar/correlate/test/CMakeLists.txt
#

set(name correlate_test)
set(${name}_SRC
    main.cpp
    Test_auto_correlate.cpp
    Test_cross_correlate.cpp
    Test_evaluate_cross_power.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)
add_test(correlate_test ${name})


if (CUDA_FOUND)
    include_directories(${CUDA_INCLUDE_DIRS})
    # Correlate kernel performance test.
    #set(name correlate_performance_test)
    #cuda_add_executable(${name} Test_correlate_performance.cu)
    #target_link_libraries(${name} oskar gtest_main ${CUDA_LIBRARIES})

    # Correlator benchmark binary.
    set(name oskar_correlator_benchmark)
    add_executable(${name} ${name}.cpp)
    target_link_libraries(${name} oskar ${CUDA_LIBRARIES})
endif()
