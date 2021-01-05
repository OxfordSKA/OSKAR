#
# oskar/apps/CMakeLists.txt
#

add_subdirectory(xml)

set(apps_SRC
    src/oskar_app_settings_fit_element_data.cpp
    src/oskar_app_settings_imager.cpp
    src/oskar_app_settings_sim_beam_pattern.cpp
    src/oskar_app_settings_sim_interferometer.cpp
    src/oskar_app_settings.cpp
    src/oskar_settings_log.cpp
    src/oskar_settings_to_beam_pattern.cpp
    src/oskar_settings_to_imager.cpp
    src/oskar_settings_to_interferometer.cpp
    src/oskar_settings_to_sky.cpp
    src/oskar_settings_to_telescope.cpp
    )
add_library(oskar_apps ${apps_SRC})
target_link_libraries(oskar_apps oskar oskar_settings)

set_target_properties(oskar_apps PROPERTIES
    SOVERSION ${OSKAR_VERSION}
    VERSION ${OSKAR_VERSION})

# Install target.
install(TARGETS oskar_apps
    ARCHIVE DESTINATION ${OSKAR_LIB_INSTALL_DIR} COMPONENT libraries
    LIBRARY DESTINATION ${OSKAR_LIB_INSTALL_DIR} COMPONENT libraries
    RUNTIME DESTINATION ${OSKAR_BIN_INSTALL_DIR} COMPONENT runtime
)

if (BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
    add_subdirectory(test)
endif()
