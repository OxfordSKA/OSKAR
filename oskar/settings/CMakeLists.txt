#
# oskar/settings/CMakeLists.txt
#

set(oskar_settings_SRC
    src/oskar_option_parser.cpp
    src/oskar_settings_utility_string.cpp
    src/oskar_SettingsDeclareXml.cpp
    src/oskar_SettingsDependency.cpp
    src/oskar_SettingsDependencyGroup.cpp
    src/oskar_SettingsFileHandler.cpp
    src/oskar_SettingsFileHandlerIni.cpp
    src/oskar_SettingsItem.cpp
    src/oskar_SettingsKey.cpp
    src/oskar_SettingsNode.cpp
    src/oskar_SettingsTree.cpp
    src/oskar_SettingsValue.cpp
)
add_subdirectory(types)
list(APPEND oskar_settings_SRC ${settings_types_SRC})

add_library(oskar_settings ${oskar_settings_SRC})
set_target_properties(oskar_settings PROPERTIES
    SOVERSION ${OSKAR_VERSION}
    VERSION ${OSKAR_VERSION})

# Install target for the settings library.
install(TARGETS oskar_settings
    ARCHIVE DESTINATION ${OSKAR_LIB_INSTALL_DIR} COMPONENT libraries
    LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR} COMPONENT libraries
    RUNTIME DESTINATION ${OSKAR_BIN_INSTALL_DIR} COMPONENT runtime
)

if (BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
    add_subdirectory(test)
endif()
