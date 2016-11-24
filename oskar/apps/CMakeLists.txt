#
# oskar/apps/CMakeLists.txt
#

set(apps_SRC
    src/oskar_beam_pattern_generate_coordinates.c
    src/oskar_set_up_sky.c
    src/oskar_set_up_telescope.c
    src/oskar_sim_beam_pattern.c
    src/oskar_sim_tec_screen.cpp
    )
add_library(oskar_apps ${apps_SRC})

target_link_libraries(oskar_apps
    oskar
    oskar_settings_containers
    oskar_settings_load
    oskar_settings_log
    ${QT_QTCORE_LIBRARY})

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

# Tests
add_subdirectory(test)
