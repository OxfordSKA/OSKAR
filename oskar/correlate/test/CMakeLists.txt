#
# oskar/correlate/test/CMakeLists.txt
#

set(name correlate_test)
set(${name}_SRC
    main.cpp
    Test_auto_correlate.cpp
    Test_cross_correlate.cpp
    Test_evaluate_auto_power.cpp
    Test_evaluate_cross_power.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)
add_test(NAME ${name} COMMAND ${name})
if (OpenMP_CXX_FOUND)
    target_link_libraries(${name} OpenMP::OpenMP_CXX)
endif()

set(name oskar_correlator_benchmark)
add_executable(${name} ${name}.cpp)
target_link_libraries(${name} oskar oskar_settings)
if (OpenMP_CXX_FOUND)
    target_link_libraries(${name} OpenMP::OpenMP_CXX)
endif()
