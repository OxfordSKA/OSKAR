#
# settings/list/test/CMakeLists.txt
#

set(name test_oskar_settings_list)
add_executable(${name}
#    Test_SettingsItem.cpp
#    Test_SettingsKey.cpp
#    Test_SettingsList.cpp
    Test_Variant.cpp
#    Test_Version.cpp
)
target_link_libraries(${name}
    gtest_main
    oskar_settings_list
)
add_test(${name} ${name})



