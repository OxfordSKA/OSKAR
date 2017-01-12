#
# gui/CMakeLists.txt
#

# Build OSKAR GUI.
find_package(Qt5Core)
find_package(Qt5Widgets)
find_package(Qt5Network)

if (Qt5Core_FOUND AND Qt5Widgets_FOUND AND Qt5Network_FOUND)
    set(gui_SRC
        src/oskar_main.cpp
        src/oskar_About.cpp
        src/oskar_AppLocation.cpp
        src/oskar_DoubleSpinBox.cpp
        src/oskar_Docs.cpp
        src/oskar_MainWindow.cpp
        src/oskar_RunDialog.cpp
        src/oskar_SettingsDelegate.cpp
        src/oskar_SettingsModel.cpp
        src/oskar_SettingsView.cpp
    )
    set(gui_MOC_SRC
        oskar_About.h
        oskar_AppLocation.h
        oskar_DoubleSpinBox.h
        oskar_Docs.h
        oskar_MainWindow.h
        oskar_RunDialog.h
        oskar_SettingsDelegate.h
        oskar_SettingsModel.h
        oskar_SettingsView.h
    )
    qt5_wrap_cpp(gui_SRC ${gui_MOC_SRC})
    qt5_add_resources(gui_SRC icons/icons.qrc)
    set(target "oskar_app")
    if (APPLE)
        set(MACOSX_BUNDLE_BUNDLE_NAME "OSKAR")
        set(MACOSX_BUNDLE_GUI_IDENTIFIER "OSKAR")
        set(MACOSX_BUNDLE_ICON_FILE oskar.icns)
        set(MACOSX_BUNDLE_INFO_STRING
            "OSKAR ${OSKAR_VERSION_MAJOR}.${OSKAR_VERSION_MINOR}.${OSKAR_VERSION_PATCH}")
        set(MACOSX_BUNDLE_SHORT_VERSION_STRING
            "${OSKAR_VERSION_MAJOR}.${OSKAR_VERSION_MINOR}")
        set(MACOSX_BUNDLE_LONG_VERSION_STRING
            "${OSKAR_VERSION_MAJOR}.${OSKAR_VERSION_MINOR}.${OSKAR_VERSION_PATCH}")
        set(MACOSX_BUNDLE_BUNDLE_VERSION
            "${OSKAR_VERSION_MAJOR}.${OSKAR_VERSION_MINOR}.${OSKAR_VERSION_PATCH}")
        set(MACOSX_BUNDLE_COPYRIGHT "(C) 2012-2017 The University of Oxford")
        set(gui_ICON icons/oskar.icns)
        set_source_files_properties(${gui_ICON} PROPERTIES
            MACOSX_PACKAGE_LOCATION "Resources")
        add_executable(${target} MACOSX_BUNDLE ${gui_ICON} ${gui_SRC})
    else (APPLE)
        add_executable(${target} ${gui_SRC})
    endif (APPLE)
    target_link_libraries(${target}
            oskar_settings
            Qt5::Core
            Qt5::Widgets
            Qt5::Network
    )
    set_target_properties(${target} PROPERTIES OUTPUT_NAME oskar)
    install(TARGETS ${target} DESTINATION ${OSKAR_BIN_INSTALL_DIR})
    if (APPLE)
        # For convenience, add a BASH script to run the binary inside
        # the .app bundle. Unfortunately a symlink doesn't do the job.
        configure_file(oskar ${CMAKE_CURRENT_BINARY_DIR}/oskar @ONLY)
        install(FILES ${CMAKE_CURRENT_BINARY_DIR}/oskar
            DESTINATION "${OSKAR_BIN_INSTALL_DIR}"
            PERMISSIONS
                OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
            )
        # Create a standalone .app bundle.
        # set(BU_CHMOD_BUNDLE_ITEMS ON)
        # include(BundleUtilities)
    endif (APPLE)
else()
    message("===============================================================================")
    message("-- WARNING: Qt 5 not found: Unable to build OSKAR GUI.")
    message("-- Specify path to Qt 5 using -DCMAKE_PREFIX_PATH if required.")
    message("--     e.g. -DCMAKE_PREFIX_PATH=/usr/local/opt/qt5/")
    message("===============================================================================")
endif ()
