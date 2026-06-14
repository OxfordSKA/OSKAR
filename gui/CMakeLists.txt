#
# gui/CMakeLists.txt
#

# Build OSKAR GUI.
find_package(Qt6 6.2 QUIET COMPONENTS Core Gui Widgets Network)
if (NOT Qt6_FOUND)
    find_package(Qt5Core QUIET)
    find_package(Qt5Widgets QUIET)
    find_package(Qt5Network QUIET)
endif()

if (Qt6_FOUND OR (Qt5Core_FOUND AND Qt5Widgets_FOUND AND Qt5Network_FOUND))
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
    if (Qt6_FOUND)
        qt6_wrap_cpp(gui_SRC ${gui_MOC_SRC})
        qt6_add_resources(gui_SRC icons/icons.qrc)
    else()
        qt5_wrap_cpp(gui_SRC ${gui_MOC_SRC})
        qt5_add_resources(gui_SRC icons/icons.qrc)
    endif()
    set(target "oskar_app")
    if (APPLE)
        add_executable(${target} ${gui_SRC})
        set_target_properties(${target}
            PROPERTIES
            OUTPUT_NAME oskar
        )
    elseif (WIN32)
        set(gui_ICON icons/oskar.rc)
        add_executable(${target} WIN32 ${gui_ICON} ${gui_SRC})
        set_target_properties(${target} PROPERTIES OUTPUT_NAME oskar)
    else ()
        add_executable(${target} ${gui_SRC})
        set_target_properties(${target} PROPERTIES OUTPUT_NAME oskar)
    endif ()
    if (Qt6_FOUND)
        target_link_libraries(${target}
                oskar_settings
                Qt6::Core
                Qt6::Widgets
                Qt6::Network
        )
        target_compile_features(${target} PRIVATE cxx_std_17)
    else()
        target_link_libraries(${target}
                oskar_settings
                Qt5::Core
                Qt5::Widgets
                Qt5::Network
        )
    endif()
    install(TARGETS ${target}
        DESTINATION ${OSKAR_BIN_INSTALL_DIR} COMPONENT applications)
    if (APPLE)
        if (Qt6_FOUND)
            get_target_property(QtMac_location Qt6::QCocoaIntegrationPlugin LOCATION)
            get_target_property(QtMacStyle_location Qt6::QMacStylePlugin LOCATION)
        else()
            get_target_property(QtMac_location Qt5::QCocoaIntegrationPlugin LOCATION)
            get_target_property(QtMacStyle_location Qt5::QMacStylePlugin LOCATION)
        endif()
        set(QtMac_location ${QtMac_location} CACHE INTERNAL "plugin location")
        set(QtMacStyle_location ${QtMacStyle_location} CACHE INTERNAL "plugin location")
    elseif (WIN32)
        if (Qt6_FOUND)
            get_target_property(QtCore_location Qt6::Core LOCATION)
            get_target_property(QtGui_location Qt6::Gui LOCATION)
            get_target_property(QtWidgets_location Qt6::Widgets LOCATION)
            get_target_property(QtNetwork_location Qt6::Network LOCATION)
            get_target_property(QtWin_location Qt6::QWindowsIntegrationPlugin LOCATION)
            # The Windows style plugin was renamed in Qt 6.7.
            if (TARGET Qt6::QModernWindowsStylePlugin)
                get_target_property(QtWinStyle_location Qt6::QModernWindowsStylePlugin LOCATION)
            elseif (TARGET Qt6::QWindowsVistaStylePlugin)
                get_target_property(QtWinStyle_location Qt6::QWindowsVistaStylePlugin LOCATION)
            endif ()
        else()
            get_target_property(QtCore_location Qt5::Core LOCATION)
            get_target_property(QtGui_location Qt5::Gui LOCATION)
            get_target_property(QtWidgets_location Qt5::Widgets LOCATION)
            get_target_property(QtNetwork_location Qt5::Network LOCATION)
            get_target_property(QtWin_location Qt5::QWindowsIntegrationPlugin LOCATION)
            get_target_property(QtWinStyle_location Qt5::QWindowsVistaStylePlugin LOCATION)
        endif()
        install(FILES ${QtCore_location} ${QtGui_location}
            ${QtWidgets_location} ${QtNetwork_location}
            DESTINATION ${OSKAR_BIN_INSTALL_DIR} COMPONENT runtime)
        install(FILES ${QtWin_location}
            DESTINATION ${OSKAR_BIN_INSTALL_DIR}/platforms COMPONENT runtime)
        install(FILES ${QtWinStyle_location}
            DESTINATION ${OSKAR_BIN_INSTALL_DIR}/styles COMPONENT runtime)
    endif ()
else()
    message("===============================================================================")
    message("-- WARNING: Qt 6 (>= 6.2) not found: Unable to build OSKAR GUI.")
    message("-- Specify path to Qt 6 using -DCMAKE_PREFIX_PATH if required.")
    message("--     e.g. -DCMAKE_PREFIX_PATH=$(brew --prefix qt6)")
    message("===============================================================================")
endif ()
