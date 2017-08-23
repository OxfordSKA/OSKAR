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
    # src/oskar_sim_tec_screen.cpp
    )
add_library(oskar_apps ${apps_SRC})

target_link_libraries(oskar_apps
    oskar
    oskar_settings)

if (MSVC)
    set_target_properties(oskar_apps PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
        INSTALL_RPATH_USE_LINK_PATH TRUE
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        SOVERSION ${OSKAR_VERSION}
        VERSION ${OSKAR_VERSION})
else ()
    set_target_properties(oskar_apps PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
        INSTALL_RPATH_USE_LINK_PATH TRUE
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        LINK_FLAGS "${OpenMP_CXX_FLAGS}"
        SOVERSION ${OSKAR_VERSION}
        VERSION ${OSKAR_VERSION})
endif()

# Install target for the apps library.
install(TARGETS oskar_apps DESTINATION ${OSKAR_LIB_INSTALL_DIR})
