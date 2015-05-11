#
# settings/types/test/CMakeLists.txt
#

set(name Test_settings_types)
file(GLOB ${name}_SRC Test_*.cpp)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name}
    gtest_main
    oskar_settings_types
)
add_test(${name} ${name})
