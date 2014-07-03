#
# src/widgets/CMakeLists.txt
#

if (QT4_FOUND)

    include_directories(${QT_INCLUDE_DIR})

    qt4_add_resources(widgets_RCC_SRC icons/icons.qrc)

    set(widgets_MOC_SRC
        oskar_About.h
        oskar_BinaryLocations.h
        oskar_CudaInfoDisplay.h
        oskar_DocumentationDisplay.h
        oskar_RunDialog.h
        oskar_RunThread.h
    )
    set(widgets_SRC
        src/oskar_About.cpp
        src/oskar_BinaryLocations.cpp
        src/oskar_CudaInfoDisplay.cpp
        src/oskar_DocumentationDisplay.cpp
        src/oskar_RunDialog.cpp
        src/oskar_RunThread.cpp
    )

    # Build and install oskar_widgets library
    qt4_wrap_cpp(widgets_SRC ${widgets_MOC_SRC})
    add_library(oskar_widgets ${widgets_SRC} ${widgets_RCC_SRC})
    target_link_libraries(oskar_widgets
        oskar_settings_widgets
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )
    set_target_properties(oskar_widgets PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
        INSTALL_RPATH_USE_LINK_PATH TRUE
        SOVERSION ${OSKAR_VERSION}
        VERSION ${OSKAR_VERSION})
    install(TARGETS oskar_widgets DESTINATION ${OSKAR_LIB_INSTALL_DIR})

endif (QT4_FOUND)
