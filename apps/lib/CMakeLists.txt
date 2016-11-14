#
# src/apps/lib/CMakeLists.txt
#

set(apps_SRC
    src/oskar_beam_pattern_generate_coordinates.c
    src/oskar_evaluate_station_pierce_points.cpp
    src/oskar_set_up_sky.c
    src/oskar_set_up_telescope.c
    src/oskar_sim_beam_pattern.c
    src/oskar_sim_interferometer.c
    src/oskar_sim_tec_screen.cpp
    )
if(CASACORE_FOUND)
    list(APPEND apps_SRC
        src/oskar_vis_block_write_ms.c
        src/oskar_vis_header_write_ms.c)
endif()
add_library(oskar_apps ${apps_SRC})

# Link in oskar_ms if avaliable.
set(settings_libs oskar_settings_containers oskar_settings_load oskar_settings_log)
if(CASACORE_FOUND)
    target_link_libraries(oskar_apps oskar oskar_ms
        ${settings_libs} ${QT_QTCORE_LIBRARY})
else()
    target_link_libraries(oskar_apps oskar
        ${settings_libs} ${QT_QTCORE_LIBRARY})
endif()

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
