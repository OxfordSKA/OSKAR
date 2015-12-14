#
# settings/containers/CMakeLists.txt
#

# FIXME-BM: Not the best place for this include.
include_directories(${QT_INCLUDE_DIR})

set(name oskar_settings_containers)
add_library(${name} STATIC
    src/oskar_SettingsDeclareXml.cpp
    src/oskar_SettingsDependency.cpp
    src/oskar_SettingsDependencyGroup.cpp
    src/oskar_SettingsFileHandlerQSettings.cpp
    src/oskar_SettingsItem.cpp
    src/oskar_SettingsKey.cpp
    src/oskar_SettingsNode.cpp
    src/oskar_SettingsTree.cpp
)
target_link_libraries(${name} oskar_settings_types ${QT_QTCORE_LIBRARY})

add_subdirectory(test)
