#
# settings/types/CMakeLists.txt
#

set(name oskar_settings_types)
add_library(${name} STATIC
#    src/DateTime.cpp
#    src/DoubleList.cpp
#    src/DoubleRange.cpp
#    src/InputDirectory.cpp
#    src/InputFile.cpp
#    src/IntList.cpp
    src/IntRange.cpp
#    src/OptionList.cpp
#    src/OutputFile.cpp
#    src/StringList.cpp
#    src/Time.cpp
)

add_subdirectory(test)

