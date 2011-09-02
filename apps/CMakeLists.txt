#
# src/apps/CMakeLists.txt
#

set(apps_SRC
    src/oskar_load_stations.cpp
    src/oskar_load_telescope.cpp
    src/oskar_Settings.cpp
    src/oskar_imager_dft.cu
)
set(apps_SRC "${apps_SRC}" PARENT_SCOPE)

# ==== Macros to build the oskar-lib platform module.
include_directories(${QT_INCLUDE_DIR})
cuda_add_library(oskar_apps STATIC ${apps_SRC})
target_link_libraries(oskar_apps ${QT_QTCORE_LIBRARY} oskar)
set_target_properties(oskar_apps PROPERTIES LINKER_LANGUAGE CXX)


# ==== Interferometer1_scalar standalone binary.
add_executable(sim1_scalar src/oskar_sim1_scalar.cpp)
target_link_libraries(sim1_scalar
    oskar
    oskar_apps
    ${QT_QTCORE_LIBRARY}
)

# ==== Interferometer1_scalar frequency scan test standalone binary.
add_executable(sim1_scalar_freq_scan src/oskar_sim1_scalar_freq_scan.cpp)
target_link_libraries(sim1_scalar_freq_scan
    oskar
    oskar_apps
    ${QT_QTCORE_LIBRARY}
)
if (BUILD_OSKAR_WIDGETS)
    add_executable(imager_scalar_freq_scan src/oskar_imager_scalar_freq_scan.cpp)
    target_link_libraries(imager_scalar_freq_scan
        oskar
        oskar_apps
        oskar_widgets
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )
endif ()


# ==== Interferometer1_scalar standalone binary.
if (BUILD_OSKAR_WIDGETS)
    include_directories(${Qwt5_INCLUDE_DIR})
    include_directories(${QT_QTGUI_INCLUDE_DIR})
    include_directories(${QT_INCLUDE_DIR})
    include_directories(${QT_QTCORE_INCLUDE_DIR})
    add_executable(imager_scalar src/oskar_imager_scalar.cpp)
    target_link_libraries(imager_scalar
        oskar
        oskar_apps
        oskar_widgets
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )
endif ()

# ==== Recurse into test subdirectory.
add_subdirectory(test)
