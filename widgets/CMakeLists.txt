#
# src/widgets/CMakeLists.txt
#

if (QT4_FOUND AND CUDA_FOUND)

    include_directories(${QT_INCLUDE_DIR})

    qt4_add_resources(widgets_RCC_SRC icons/icons.qrc)

    set(widgets_MOC_SRC
        oskar_About.h
        oskar_BinaryLocations.h
        oskar_CudaInfoDisplay.h
        oskar_DataFileModel.h
        oskar_DocumentationDisplay.h
        oskar_DoubleSpinBox.h
        oskar_RunDialog.h
        oskar_RunThread.h
        oskar_SettingsDelegate.h
        oskar_SettingsModel.h
        oskar_SettingsModelApps.h
        oskar_SettingsView.h
    )
    set(widgets_SRC
        src/oskar_About.cpp
        src/oskar_BinaryLocations.cpp
        src/oskar_CudaInfoDisplay.cpp
        src/oskar_DataFileModel.cpp
        src/oskar_DocumentationDisplay.cpp
        src/oskar_DoubleSpinBox.cpp
        src/oskar_RunDialog.cpp
        src/oskar_RunThread.cpp
        src/oskar_SettingsDelegate.cpp
        src/oskar_SettingsItem.cpp
        src/oskar_SettingsModel.cpp
        src/oskar_SettingsModelApps.cpp
        src/oskar_SettingsView.cpp
    )

    # Build and install oskar_widgets library
    qt4_wrap_cpp(widgets_SRC ${widgets_MOC_SRC})
    add_library(oskar_widgets ${widgets_SRC} ${widgets_RCC_SRC})
    target_link_libraries(oskar_widgets
        oskar
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY})
    set_target_properties(oskar_widgets PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
        INSTALL_RPATH_USE_LINK_PATH TRUE
        SOVERSION ${OSKAR_VERSION})
    install(TARGETS oskar_widgets DESTINATION ${OSKAR_LIB_INSTALL_DIR})

    # Tests
    add_subdirectory(test)

endif (QT4_FOUND AND CUDA_FOUND)
