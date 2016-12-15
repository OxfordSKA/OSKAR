#
# oskar/beam_pattern/test/CMakeLists.txt
#

set(name test_beam_pattern_coordinates)
add_executable(${name}
    Test_beam_pattern_coordinates.cpp)
target_link_libraries(${name} oskar gtest_main)
