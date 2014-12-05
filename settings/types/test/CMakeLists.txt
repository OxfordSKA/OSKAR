#
# types/test/CMakeLists.txt
#

set(name test_settings_types)
add_executable(${name}
    Test_Types.cpp
)
target_link_libraries(${name}
    gtest_main
    oskar_settings_types
)
add_test(${name} ${name})
