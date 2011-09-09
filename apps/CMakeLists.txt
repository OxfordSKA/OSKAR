#
# src/apps/CMakeLists.txt
#

# Recurse into lib subdirectory.
add_subdirectory(lib)


# BINARY: sim1_scalar_d
add_executable(sim1_scalar_d sim1_scalar_d.cpp)
target_link_libraries(sim1_scalar_d
    oskar
    oskar_apps
    ${QT_QTCORE_LIBRARY}
)

# BINARY: sim1_scalar_f
add_executable(sim1_scalar_f sim1_scalar_f.cpp)
target_link_libraries(sim1_scalar_f
    oskar
    oskar_apps
    ${QT_QTCORE_LIBRARY}
)


# BINARY: sim1_scalar_frequency_scan
add_executable(sim1_scalar_freq_scan sim1_scalar_freq_scan.cpp)
target_link_libraries(sim1_scalar_freq_scan
    oskar
    oskar_apps
    ${QT_QTCORE_LIBRARY}
)


if (BUILD_OSKAR_WIDGETS)
    include_directories(${Qwt5_INCLUDE_DIR})
    include_directories(${QT_QTGUI_INCLUDE_DIR})
    include_directories(${QT_INCLUDE_DIR})
    include_directories(${QT_QTCORE_INCLUDE_DIR})

    # BINARY: imager_scalar_d
    add_executable(imager_scalar_d imager_scalar_d.cpp)
    target_link_libraries(imager_scalar_d
        oskar
        oskar_apps
        oskar_widgets
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )

    # BINARY: imager_scalar_f
    add_executable(imager_scalar_f imager_scalar_f.cpp)
    target_link_libraries(imager_scalar_f
        oskar
        oskar_apps
        oskar_widgets
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )

    # BINARY: imager_scalar_freq_scan
    add_executable(imager_scalar_freq_scan imager_scalar_freq_scan.cpp)
    target_link_libraries(imager_scalar_freq_scan
        oskar
        oskar_apps
        oskar_widgets
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )
endif ()


