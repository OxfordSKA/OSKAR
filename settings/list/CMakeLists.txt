#
# settings/list/CMakeLists.txt
#

set(name oskar_settings_list)
add_library(${name} STATIC
    src/Variant.cpp
#    src/Version.cpp
#    src/SettingsKey.cpp
#    src/SettingsList.cpp
#    src/SettingsItem.cpp
)
target_link_libraries(${name} oskar_settings_types)

add_subdirectory(test)
