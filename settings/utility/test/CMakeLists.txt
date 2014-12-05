#
# settings/utility/test/CMakeLists.txt
#

set(name test_settings_utility)
add_executable(${name}
    Test_oskar_settings_utility_string
)
target_link_libraries(${name}
    gtest_main
    oskar_settings_utility
)

