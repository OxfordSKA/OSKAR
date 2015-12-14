#
# settings/list/test/CMakeLists.txt
#

set(name Test_SettingsList)
add_executable(${name}
#    Test_SettingsItem.cpp
    Test_SettingsKey.cpp
#    Test_Dependency.cpp
#    Test_SettingsList.cpp
#    Test_Variant.cpp
)
target_link_libraries(${name}
    gtest_main
    oskar_settings_list
)
add_test(${name} ${name})



