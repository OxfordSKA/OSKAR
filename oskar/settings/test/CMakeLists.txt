#
# settings/test/CMakeLists.txt
#

set(name settings_test)
add_executable(${name}
    Test_Bool.cpp
    Test_DateTime.cpp
    Test_Double.cpp
    Test_DoubleList.cpp
    Test_DoubleRange.cpp
    Test_DoubleRangeExt.cpp
    Test_InputDirectory.cpp
    Test_InputFile.cpp
    Test_InputFileList.cpp
    Test_Int.cpp
    Test_IntList.cpp
    Test_IntListExt.cpp
    Test_IntPositive.cpp
    Test_IntRange.cpp
    Test_IntRangeExt.cpp
    Test_OptionList.cpp
    Test_OutputFile.cpp
    Test_RandomSeed.cpp
    Test_settings_utility_string.cpp
    Test_SettingsDependency.cpp
    Test_SettingsDependencyGroup.cpp
    Test_SettingsItem.cpp
    Test_SettingsKey.cpp
    Test_SettingsNode.cpp
    Test_SettingsTree.cpp
    Test_SettingsValue.cpp
    Test_String.cpp
    Test_StringList.cpp
    Test_Time.cpp
    Test_UnsignedInt.cpp
    Test_UnsignedDouble.cpp
)
target_link_libraries(${name}
    gtest_main
    oskar_settings
)
add_test(${name} ${name})
