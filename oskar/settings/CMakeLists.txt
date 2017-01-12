#
# settings/CMakeLists.txt
#

set(oskar_settings_SRC
    src/oskar_settings_utility_string.cpp
    src/oskar_SettingsDeclareXml.cpp
    src/oskar_SettingsDependency.cpp
    src/oskar_SettingsDependencyGroup.cpp
    src/oskar_SettingsFileHandlerIni.cpp
    src/oskar_SettingsItem.cpp
    src/oskar_SettingsKey.cpp
    src/oskar_SettingsNode.cpp
    src/oskar_SettingsTree.cpp
    src/oskar_SettingsValue.cpp
)
add_subdirectory(types)
list(APPEND oskar_settings_SRC ${settings_types_SRC})

# This doesn't have to be static, but it's more convenient for the GUI.
add_library(oskar_settings STATIC ${oskar_settings_SRC})
set_target_properties(oskar_settings PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
    INSTALL_RPATH_USE_LINK_PATH TRUE
    SOVERSION ${OSKAR_VERSION}
    VERSION ${OSKAR_VERSION})

# Install target for the settings library.
install(TARGETS oskar_settings DESTINATION ${OSKAR_LIB_INSTALL_DIR})

add_subdirectory(test)
