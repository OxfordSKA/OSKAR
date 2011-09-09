#
# src/apps/CMakeLists.txt
#

# Recurse into lib subdirectory.
add_subdirectory(lib)


# BINARY: sim1_scalar_d
add_executable(oskar_sim1_scalar_d oskar_sim1_scalar_d.cpp)
target_link_libraries(oskar_sim1_scalar_d oskar oskar_apps ${QT_QTCORE_LIBRARY})
install(TARGETS oskar_sim1_scalar_d DESTINATION ${OSKAR_BIN_INSTALL_DIR})



# BINARY: sim1_scalar_f
add_executable(oskar_sim1_scalar_f oskar_sim1_scalar_f.cpp)
target_link_libraries(oskar_sim1_scalar_f oskar oskar_apps ${QT_QTCORE_LIBRARY})
install(TARGETS oskar_sim1_scalar_f DESTINATION ${OSKAR_BIN_INSTALL_DIR})


# BINARY: sim1_scalar_frequency_scan
add_executable(oskar_sim1_scalar_freq_scan oskar_sim1_scalar_freq_scan.cpp)
target_link_libraries(oskar_sim1_scalar_freq_scan oskar oskar_apps ${QT_QTCORE_LIBRARY})


if (BUILD_OSKAR_WIDGETS)
    include_directories(${Qwt5_INCLUDE_DIR})
    include_directories(${QT_QTGUI_INCLUDE_DIR})
    include_directories(${QT_INCLUDE_DIR})
    include_directories(${QT_QTCORE_INCLUDE_DIR})

    # BINARY: imager_scalar_d
    add_executable(oskar_imager_scalar_d oskar_imager_scalar_d.cpp)
    target_link_libraries(oskar_imager_scalar_d oskar oskar_apps oskar_widgets
        ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY})
    install(TARGETS oskar_imager_scalar_d DESTINATION ${OSKAR_BIN_INSTALL_DIR})

    # BINARY: imager_scalar_f
    add_executable(oskar_imager_scalar_f oskar_imager_scalar_f.cpp)
    target_link_libraries(oskar_imager_scalar_f oskar oskar_apps oskar_widgets
        ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY})
    install(TARGETS oskar_imager_scalar_f DESTINATION ${OSKAR_BIN_INSTALL_DIR})

    # BINARY: imager_scalar_freq_scan
    add_executable(oskar_imager_scalar_freq_scan oskar_imager_scalar_freq_scan.cpp)
    target_link_libraries(oskar_imager_scalar_freq_scan oskar oskar_apps
        oskar_widgets ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY})

endif ()


