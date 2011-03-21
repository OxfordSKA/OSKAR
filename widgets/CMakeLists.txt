#
# src/widgets/CMakeLists.txt
#


if (QtGui_FOUND)

    set(widgets_MOC_SRC
        #
    )

    set(widgets_UI_FORMS
        #
    )

    set(widgets_SRC
        #
    )

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



if (Qwt5_FOUND)
    add_subdirectory(plotting)
endif()

install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DESTINATION ${OSKAR_INCLUDE_INSTALL_DIR}
    FILES_MATCHING PATTERN "*.h"
    PATTERN "src" EXCLUDE
    PATTERN "test" EXCLUDE)

add_subdirectory(test)
