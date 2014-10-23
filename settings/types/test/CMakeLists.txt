#
# types/test/CMakeLists.txt
#

set(name test_settings_types)
add_executable(${name}
#    Test_DateTime.cpp
#    Test_DoubleList.cpp
    Test_DoubleRange.cpp
#    Test_InputDirectory.cpp
#    Test_InputFile.cpp
#    Test_IntRange.cpp
#    Test_OptionList.cpp
#    Test_OutputFile.cpp
#    Test_StringList.cpp
)
target_link_libraries(${name}
    gtest_main
    oskar_settings_types
)
add_test(${name} ${name})
