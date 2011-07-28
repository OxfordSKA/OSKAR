#
# src/widgets/CMakeLists.txt
#


# ==== Plotting widgets sub-module.
if (Qwt5_FOUND)

    add_subdirectory(plotting)

    # ==== Concatenate source code for widget subfolders.
    foreach (file ${widgets_plotting_SRC})
        list(APPEND widgets_SRC plotting/${file})
    endforeach ()
    foreach (file ${widgets_plotting_MOC})
        list(APPEND widgets_MOC_SRC plotting/${file})
    endforeach ()
    qt4_wrap_cpp(widgets_SRC ${widgets_MOC_SRC})

    # ==== Build the widgets library.
    include_directories(${Qwt5_INCLUDE_DIR})
    include_directories(${QT_INCLUDE_DIR})
    include_directories(${QT_QTCORE_INCLUDE_DIR})
    include_directories(${QT_QTGUI_INCLUDE_DIR})

    add_library(oskar_widgets SHARED ${widgets_SRC})
    target_link_libraries(oskar_widgets
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY}
        ${Qwt5_Qt4_LIBRARY})
    set_target_properties(oskar_widgets PROPERTIES
        INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib
        INSTALL_RPATH_USE_LINK_PATH TRUE)

    # ==== Install target for plotting widgets sub-module.
    set_target_properties(oskar_widgets PROPERTIES
        SOVERSION ${oskar-lib_SOVERSION})
    install(TARGETS oskar_widgets DESTINATION ${OSKAR_LIB_INSTALL_DIR})

endif()





