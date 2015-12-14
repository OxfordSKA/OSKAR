#
# settings/list/CMakeLists.txt
#

set(name oskar_settings_list)
add_library(${name} STATIC
#    src/oskar_SettingsVariant.cpp
    src/oskar_SettingsKey.cpp
#    src/oskar_SettingsList.cpp
#    src/oskar_SettingsItem.cpp
)
target_link_libraries(${name} oskar_settings_types)

add_subdirectory(test)
