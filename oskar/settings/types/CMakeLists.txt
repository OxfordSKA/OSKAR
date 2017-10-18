#
# settings/types/CMakeLists.txt
#

set(settings_types_SRC
    types/src/oskar_AbstractSettingsType.cpp
    types/src/oskar_Bool.cpp
    types/src/oskar_DateTime.cpp
    types/src/oskar_Double.cpp
    types/src/oskar_DoubleList.cpp
    types/src/oskar_DoubleRange.cpp
    types/src/oskar_DoubleRangeExt.cpp
    types/src/oskar_InputDirectory.cpp
    types/src/oskar_InputFile.cpp
    types/src/oskar_InputFileList.cpp
    types/src/oskar_Int.cpp
    types/src/oskar_IntList.cpp
    types/src/oskar_IntListExt.cpp
    types/src/oskar_IntPositive.cpp
    types/src/oskar_IntRange.cpp
    types/src/oskar_IntRangeExt.cpp
    types/src/oskar_OptionList.cpp
    types/src/oskar_OutputFile.cpp
    types/src/oskar_RandomSeed.cpp
    types/src/oskar_String.cpp
    types/src/oskar_StringList.cpp
    types/src/oskar_Time.cpp
    types/src/oskar_UnsignedDouble.cpp
    types/src/oskar_UnsignedInt.cpp
)

set(settings_types_SRC "${settings_types_SRC}" PARENT_SCOPE)
