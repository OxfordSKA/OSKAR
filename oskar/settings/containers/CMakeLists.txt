#
# settings/containers/CMakeLists.txt
#

set(name oskar_settings_containers)
add_library(${name} STATIC
    src/oskar_SettingsDeclareXml.cpp
    src/oskar_SettingsDependency.cpp
    src/oskar_SettingsDependencyGroup.cpp
    src/oskar_SettingsFileHandlerIni.cpp
    src/oskar_SettingsItem.cpp
    src/oskar_SettingsKey.cpp
    src/oskar_SettingsNode.cpp
    src/oskar_SettingsTree.cpp
)
target_link_libraries(${name} oskar_settings_types)

add_subdirectory(test)
