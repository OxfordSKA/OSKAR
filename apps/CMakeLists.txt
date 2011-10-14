#
# src/apps/CMakeLists.txt
#

# Recurse into lib subdirectory.
add_subdirectory(lib)

# BINARY: oskar_sim1_scalar
add_executable(oskar_sim1_scalar oskar_sim1_scalar.cpp)
target_link_libraries(oskar_sim1_scalar oskar oskar_apps ${QT_QTCORE_LIBRARY})
set_target_properties(oskar_sim1_scalar PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
#install(TARGETS oskar_sim1_scalar DESTINATION ${OSKAR_BIN_INSTALL_DIR})

if (BUILD_OSKAR_WIDGETS)
    include_directories(${Qwt5_INCLUDE_DIR})
    include_directories(${QT_QTGUI_INCLUDE_DIR})
    include_directories(${QT_INCLUDE_DIR})
    include_directories(${QT_QTCORE_INCLUDE_DIR})

    # BINARY: oskar_imager_scalar
    add_executable(oskar_imager_scalar oskar_imager_scalar.cpp)
    target_link_libraries(oskar_imager_scalar oskar oskar_apps oskar_widgets
        ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY})
    set_target_properties(oskar_imager_scalar PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
    #install(TARGETS oskar_imager_scalar DESTINATION ${OSKAR_BIN_INSTALL_DIR})

    # BINARY: oskar_viewer_sclar
    add_executable(oskar_viewer_scalar oskar_viewer_scalar.cpp)
    target_link_libraries(oskar_viewer_scalar oskar oskar_apps oskar_widgets
        ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY})
    set_target_properties(oskar_viewer_scalar PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
    #install(TARGETS oskar_viewer_scalar DESTINATION ${OSKAR_BIN_INSTALL_DIR})


endif ()


