#
# src/apps/CMakeLists.txt
#

add_subdirectory(test)
add_subdirectory(log)

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
            ${QT_QTCORE_LIBRARY}
            ${QT_QTGUI_LIBRARY}
            ${QT_QTNETWORK_LIBRARY})
endif ()

if (CUDA_FOUND)
    macro(declare_oskar_apps)
        foreach (name ${ARGN})
            oskar_app(${name} ${name}_main.cpp)
        endforeach()
    endmacro()

    declare_oskar_apps(
        oskar_cuda_system_info
        oskar_binary_file_query
        oskar_filter_sky_model_clusters
        oskar_vis_add
        oskar_vis_summary
        oskar_vis_upgrade_format
        oskar_vis_to_ascii_table)

    set(name oskar_fits_image_to_sky_model)
    oskar_app(${name} ${name}_main.cpp EXTRA_LIBS oskar_fits)
endif ()

if (QT4_FOUND AND CUDA_FOUND)
    # Build oskar_app library
    add_subdirectory(lib)

    # BINARY: oskar_sim_interferometer
    set(name oskar_sim_interferometer)
    oskar_qt_app(${name} ${name}_main.cpp)

    # BINARY: oskar_sim_beam_pattern
    set(name oskar_sim_beam_pattern)
    oskar_qt_app(${name} ${name}_main.cpp)

    # BINARY: oskar_imager
    set(name oskar_imager)
    oskar_qt_app(${name} ${name}_main.cpp)

    # BINARY: oskar_fit_element_data
    set(name oskar_fit_element_data)
    oskar_qt_app(${name} ${name}_main.cpp)

    # BINARY: oskar_vis_to_ms
    set(name oskar_vis_to_ms)
    oskar_qt_app(${name} ${name}_main.cpp)

    # BINARY: oskar_vis_add_noise
    set(name oskar_vis_add_noise)
    oskar_qt_app(${name} ${name}_main.cpp)

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
    PATTERN "log" EXCLUDE
    PATTERN "gui" EXCLUDE
    PATTERN "lib" EXCLUDE
    PATTERN "test" EXCLUDE
)

#if (CUDA_FOUND AND PNG_FOUND)
#    include_directories(${PNG_INCLUDE_DIRS})
#
#    set(name oskar_image_to_png)
#    oskar_app(${name} ${name}_main.cpp
#        EXTRA_LIBS ${PNG_LIBRARY}
#        NO_INSTALL)
#endif ()

#if (CUDA_FOUND AND PNG_FOUND AND QT4_FOUND)
#    include_directories(${PNG_INCLUDE_DIRS})
#    set(name oskar_ionosphere_to_kml)
#    oskar_app(${name} ${name}_main.cpp
#        EXTRA_LIBS ${PNG_LIBRARY} oskar_apps
#        NO_INSTALL
#    )
#endif ()
