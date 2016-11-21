#
# settings/list/test/CMakeLists.txt
#

set(name Test_settings_containers)
add_executable(${name}
    Test_SettingsDependency.cpp
    Test_SettingsDependencyGroup.cpp
    Test_SettingsItem.cpp
    Test_SettingsKey.cpp
    Test_SettingsNode.cpp
    Test_SettingsTree.cpp
)
target_link_libraries(${name}
    gtest_main
    oskar_settings_containers
)
add_test(${name} ${name})
