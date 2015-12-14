#
# settings/types/CMakeLists.txt
#

set(name oskar_settings_types)
# file(GLOB ${name}_SRC src/*.cpp)
set(${name}_SRC
    src/oskar_Bool.cpp
    src/oskar_DateTime.cpp
    src/oskar_Double.cpp
    src/oskar_DoubleList.cpp
    src/oskar_DoubleRange.cpp
    src/oskar_DoubleRangeExt.cpp
    src/oskar_InputDirectory.cpp
    src/oskar_InputFile.cpp
    src/oskar_InputFileList.cpp
    src/oskar_Int.cpp
    src/oskar_IntList.cpp
    src/oskar_IntListExt.cpp
    src/oskar_IntPositive.cpp
    src/oskar_IntRange.cpp
    src/oskar_IntRangeExt.cpp
    src/oskar_OptionList.cpp
    src/oskar_OutputFile.cpp
    src/oskar_RandomSeed.cpp
    src/oskar_String.cpp
    src/oskar_StringList.cpp
    src/oskar_Time.cpp
    src/oskar_UnsignedDouble.cpp
    src/oskar_UnsignedInt.cpp
    
    src/oskar_SettingsValue.cpp
)

add_library(${name} STATIC ${${name}_SRC})
target_link_libraries(${name} oskar_settings_utility)

add_subdirectory(test)

