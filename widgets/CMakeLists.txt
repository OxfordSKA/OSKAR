#
# src/widgets/CMakeLists.txt
#


set(widgets_MOC_SRC
)

set(widgets_UI_FORMS
)

set(widgets_SRC
)

if (QtGui_FOUND)
    # Include the current binary directory for the purposes of compiled forms.
    include_directories(${CMAKE_CURRENT_BINARY_DIR})

    # Compile UI forms and MOC headers.
    qt4_wrap_cpp(widgets_MOC ${widgets_MOC_SRC})
    qt4_wrap_ui(widgets_UI_H ${WIDGETS_UI_FORMS})

    # Create the widgets library.
    add_library(widgets ${widgets_SRC} ${widgets_UI_H} ${widgets_MOC})

    # Link the widgets library with dependencies.
    target_link_libraries(widgets
        utility
        data
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )

    # Recurse into test subdirectories.
    add_subdirectory(delegates)
endif()



add_subdirectory(test)
