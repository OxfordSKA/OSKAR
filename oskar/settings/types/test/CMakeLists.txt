#
# settings/types/test/CMakeLists.txt
#

set(name Test_settings_types)
# file(GLOB ${name}_SRC Test_*.cpp)
set(${name}_SRC
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
    Test_String.cpp
    Test_StringList.cpp
    Test_Time.cpp
    Test_UnsignedInt.cpp
    Test_UnsignedDouble.cpp
    Test_SettingsValue.cpp
)
add_executable(${name} ${${name}_SRC})
target_link_libraries(${name}
    gtest_main
    oskar_settings_types
)
add_test(${name} ${name})
