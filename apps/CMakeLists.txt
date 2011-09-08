#
# src/apps/CMakeLists.txt
#

#***** oskar_apps library target ***********************************************
set(apps_SRC
    src/oskar_load_stations.cpp
    src/oskar_load_telescope.cpp
    src/oskar_Settings.cpp
    src/oskar_imager_dft.cu
    src/oskar_write_ms.cpp
    src/oskar_file_utils.cpp
)
set(apps_SRC "${apps_SRC}" PARENT_SCOPE)

# ==== Macros to build the oskar-lib platform module.
include_directories(${QT_INCLUDE_DIR})
cuda_add_library(oskar_apps SHARED ${apps_SRC})
target_link_libraries(oskar_apps ${QT_QTCORE_LIBRARY} oskar oskar_ms)
set_target_properties(oskar_apps PROPERTIES LINKER_LANGUAGE CXX)
set_target_properties(oskar_apps PROPERTIES COMPILE_FLAGS "-Wno-cast-qual")
set_target_properties(oskar_apps PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
target_link_libraries(oskar_apps ${CASACORE_LIBRARIES})

# ==== Install configuration for the oskar_apps library.
install(TARGETS oskar_apps DESTINATION ${OSKAR_LIB_INSTALL_DIR})
set_target_properties(oskar_apps PROPERTIES SOVERSION ${oskar-lib_SOVERSION})
#*******************************************************************************




# BINARY: sim1_scalar
add_executable(sim1_scalar src/oskar_sim1_scalar.cpp)
target_link_libraries(sim1_scalar
    oskar
    oskar_apps
    ${QT_QTCORE_LIBRARY}
)

# BINARY: sim1_scalar_frequency_scan
add_executable(sim1_scalar_freq_scan src/oskar_sim1_scalar_freq_scan.cpp)
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

    # BINARY: imager_scalar
    add_executable(imager_scalar src/oskar_imager_scalar.cpp)
    target_link_libraries(imager_scalar
        oskar
        oskar_apps
        oskar_widgets
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )

    # BINARY: imager_scalar_freq_scan
    add_executable(imager_scalar_freq_scan src/oskar_imager_scalar_freq_scan.cpp)
    target_link_libraries(imager_scalar_freq_scan
        oskar
        oskar_apps
        oskar_widgets
        ${QT_QTCORE_LIBRARY}
        ${QT_QTGUI_LIBRARY}
    )
endif ()

# ==== Recurse into test subdirectory.
add_subdirectory(test)
