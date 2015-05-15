#
# src/apps/lib/CMakeLists.txt
#

set(apps_SRC
    src/oskar_beam_pattern_generate_coordinates.c
    src/oskar_dir.cpp
    src/oskar_evaluate_station_pierce_points.cpp
    src/oskar_fit_element_data.cpp
    src/oskar_imager.cpp
    src/oskar_set_up_sky.c
    src/oskar_set_up_telescope.c
    src/oskar_set_up_vis.c
    src/oskar_sim_beam_pattern.c
    src/oskar_sim_interferometer.c
    src/oskar_sim_tec_screen.cpp
    src/oskar_telescope_load.cpp
    src/oskar_telescope_save.cpp
    src/private_TelescopeLoadApodisation.cpp
    src/private_TelescopeLoadElementPattern.cpp
    #src/private_TelescopeLoadElementTaper.cpp
    src/private_TelescopeLoadElementTypes.cpp
    src/private_TelescopeLoadFeedAngle.cpp
    src/private_TelescopeLoadGainPhase.cpp
    src/private_TelescopeLoadLayout.cpp
    src/private_TelescopeLoadMountTypes.cpp
    src/private_TelescopeLoadNoise.cpp
    src/private_TelescopeLoadPermittedBeams.cpp)
if(CASACORE_FOUND)
    list(APPEND apps_SRC
        src/oskar_vis_block_write_ms.c
        src/oskar_vis_header_write_ms.c)
endif()

set(apps_SRC "${apps_SRC}" PARENT_SCOPE)

# ==== Macros to build the OSKAR apps module.
set(CUDA_GENERATED_OUTPUT_DIR
    ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/oskar_apps.dir/src)
cuda_add_library(oskar_apps ${apps_SRC})

# Link in oskar_ms and oskar_fits if avaliable.
set(settings_libs oskar_settings_load oskar_settings_log)
if(CASACORE_FOUND)
    target_link_libraries(oskar_apps oskar oskar_ms oskar_fits
        ${settings_libs} ${CASACORE_LIBRARIES} ${QT_QTCORE_LIBRARY}
        ${CUDA_LIBRARIES})
else()
    target_link_libraries(oskar_apps oskar oskar_fits ${settings_libs}
        ${QT_QTCORE_LIBRARY} ${CUDA_LIBRARIES})
endif()

if (MSVC)
    set_target_properties(oskar_apps PROPERTIES
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
        INSTALL_RPATH_USE_LINK_PATH TRUE
        SOVERSION ${OSKAR_VERSION}
        VERSION ${OSKAR_VERSION})
else ()
    if (APPLE)
        set_target_properties(oskar_apps PROPERTIES
            INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
            INSTALL_RPATH_USE_LINK_PATH TRUE
            COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
            LINK_FLAGS "${OpenMP_CXX_FLAGS}"
            SOVERSION ${OSKAR_VERSION}
            VERSION ${OSKAR_VERSION})
    else()
        set_target_properties(oskar_apps PROPERTIES
            INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
            INSTALL_RPATH_USE_LINK_PATH TRUE
            COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
            LINK_FLAGS "${OpenMP_CXX_FLAGS}"
            SOVERSION ${OSKAR_VERSION}
            VERSION ${OSKAR_VERSION})
    endif()
endif()

# Install target for the apps library.
install(TARGETS oskar_apps DESTINATION ${OSKAR_LIB_INSTALL_DIR})

# Tests
add_subdirectory(test)
