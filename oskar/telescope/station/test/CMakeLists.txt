#
# oskar/telescope/station/test/CMakeLists.txt
#

set(name station_test)
set(${name}_SRC
    main.cpp
    Test_element_weights_errors.cpp
    Test_evaluate_array_pattern.cpp
    Test_evaluate_jones_E.cpp
    Test_evaluate_station_beam.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name} oskar gtest)
add_test(station_test ${name})

# Array pattern benchmark binary.
set(name oskar_array_pattern_benchmark)
add_executable(${name} ${name}.cpp)
target_link_libraries(${name} oskar oskar_settings)
