#
# settings/utility/CMakeLists.txt
#

set(name oskar_settings_utility)
add_library(${name} STATIC
    src/oskar_settings_utility_string.cpp
)

add_subdirectory(test)

