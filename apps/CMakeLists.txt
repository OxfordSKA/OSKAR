#
# apps/CMakeLists.txt
#

add_subdirectory(test)
add_subdirectory(xml)

# === Include build macros used for apps and mex functions.
include(${OSKAR_SOURCE_DIR}/cmake/oskar_build_macros.cmake)

if (QT4_FOUND)
    include_directories(${QT_INCLUDE_DIR})

    # BINARY: oskar_settings_set
    set(name oskar_settings_set)
    qt_app(${name} ${name}_main.cpp)

    # BINARY: oskar_settings_get
    set(name oskar_settings_get)
    qt_app(${name} ${name}_main.cpp)

    # BINARY: oskar (GUI)
    add_subdirectory(gui)
    foreach (file ${gui_SRC})
        list(REMOVE_ITEM gui_SRC ${file})
        list(APPEND gui_SRC gui/${file})
    endforeach()
    foreach (file ${gui_MOC_SRC})
        list(REMOVE_ITEM gui_MOC_SRC ${file})
        list(APPEND gui_MOC_SRC gui/${file})
    endforeach()
    qt4_wrap_cpp(gui_SRC ${gui_MOC_SRC})
    qt4_add_resources(gui_SRC gui/icons/icons.qrc)
    qt_app(oskar oskar_main.cpp ${gui_SRC} ${gui_MOC_SRC}
        EXTRA_LIBS
            oskar_settings_widgets
            oskar_settings_types
            oskar_settings_containers
            ${QT_QTCORE_LIBRARY}
            ${QT_QTGUI_LIBRARY}
            ${QT_QTNETWORK_LIBRARY})
endif ()

macro(declare_oskar_apps)
    foreach (name ${ARGN})
        oskar_app(${name} ${name}_main.cpp)
    endforeach()
endmacro()

declare_oskar_apps(
    oskar_convert_ecef_to_enu
    oskar_convert_geodetic_to_ecef
    oskar_cuda_system_info
    oskar_binary_file_query
    oskar_filter_sky_model_clusters
    oskar_fit_element_data
    oskar_fits_image_to_sky_model
    oskar_imager
    oskar_sim_beam_pattern
    oskar_sim_interferometer
    oskar_vis_add
    oskar_vis_add_noise
    oskar_vis_summary
    oskar_vis_to_ms
    oskar_vis_upgrade_format
    oskar_vis_to_ascii_table)

if (QT4_FOUND)
    set(IONOSPHERE_TESTING OFF)
    if (IONOSPHERE_TESTING)
        # BINARY: oskar_sim_tec_screen
        set(name oskar_sim_tec_screen)
        oskar_app(${name} ${name}_main.cpp
            EXTRA_LIBS oskar_apps
            NO_INSTALL)

        # BINARY: oskar_evaluate_pierce_points
        set(name oskar_evaluate_pierce_points)
        oskar_app(${name} ${name}_main.cpp
            EXTRA_LIBS oskar_apps
            NO_INSTALL)

        #if (PNG_FOUND)
        #    include_directories(${PNG_INCLUDE_DIRS})
        #    set(name oskar_ionosphere_to_kml)
        #    oskar_app(${name} ${name}_main.cpp
        #        EXTRA_LIBS ${PNG_LIBRARY} oskar_apps
        #        NO_INSTALL
        #    )
        #endif ()
    endif()
endif ()

# Install python scripts to the installation binary directory
include(oskar_copy_files)
copy_files(${CMAKE_CURRENT_SOURCE_DIR} *.py ${CMAKE_CURRENT_BINARY_DIR})
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
    DESTINATION "${OSKAR_BIN_INSTALL_DIR}"
    FILES_MATCHING PATTERN "*.py"
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "gui" EXCLUDE
    PATTERN "test" EXCLUDE
    PATTERN "xml" EXCLUDE
)
