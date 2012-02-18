#
# src/widgets/CMakeLists.txt
#

# ==== Plotting widgets sub-module.
if (Qwt5_FOUND)
    include_directories(${Qwt5_INCLUDE_DIR})
    include_directories(${QT_INCLUDE_DIR})
    include_directories(${QT_QTCORE_INCLUDE_DIR})
    include_directories(${QT_QTGUI_INCLUDE_DIR})

    # Add plotting widgets.
    add_subdirectory(plotting)

    set(widgets_MOC_SRC
        oskar_DoubleSpinBox.h
        oskar_SettingsDelegate.h
        oskar_SettingsModel.h
        oskar_SettingsView.h
    )
    set(widgets_SRC
        src/oskar_DoubleSpinBox.cpp
        src/oskar_SettingsDelegate.cpp
        src/oskar_SettingsItem.cpp
        src/oskar_SettingsModel.cpp
        src/oskar_SettingsView.cpp
    )

    # Concatenate source code for widget subfolders.
    foreach (file ${widgets_plotting_SRC})
        list(APPEND widgets_SRC plotting/${file})
    endforeach ()
    foreach (file ${widgets_plotting_MOC})
        list(APPEND widgets_MOC_SRC plotting/${file})
    endforeach ()
    qt4_wrap_cpp(widgets_SRC ${widgets_MOC_SRC})
    add_library(oskar_widgets ${widgets_SRC})
    target_link_libraries(oskar_widgets
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY}
        ${Qwt5_Qt4_LIBRARY})
    set_target_properties(oskar_widgets
        PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
        INSTALL_RPATH_USE_LINK_PATH TRUE
        SOVERSION ${oskar-lib_SOVERSION})
    # Install target for plotting widgets sub-module.
    install(TARGETS oskar_widgets DESTINATION ${OSKAR_LIB_INSTALL_DIR})

    # Tests
    add_subdirectory(test)

endif()


