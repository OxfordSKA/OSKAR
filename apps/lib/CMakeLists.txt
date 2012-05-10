#
# src/apps/lib/CMakeLists.txt
#

set(apps_SRC
    src/oskar_imager.cpp
    src/oskar_remove_dir.cpp
    src/oskar_RunThread.cpp
    src/oskar_set_up_sky.cpp
    src/oskar_set_up_telescope.c
    src/oskar_set_up_visibilities.c
    src/oskar_settings_load_beam_pattern.cpp
    src/oskar_settings_load_image.cpp
    src/oskar_settings_load_observation.cpp
    src/oskar_settings_load_simulator.cpp
    src/oskar_settings_load_sky.cpp
    src/oskar_settings_load_telescope.cpp
    src/oskar_settings_load.c
    src/oskar_SettingsModelApps.cpp
    src/oskar_sim_beam_pattern.cpp
    src/oskar_sim_interferometer.cpp
    src/oskar_telescope_model_load.cpp
    src/oskar_telescope_model_save.cpp
)
if(CASACORE_FOUND)
    list(APPEND apps_SRC src/oskar_write_ms.cpp)
endif()

set(apps_MOC_SRC
    oskar_SettingsModelApps.h
    oskar_RunThread.h
)

qt4_wrap_cpp(apps_SRC ${apps_MOC_SRC})

set(apps_SRC "${apps_SRC}" PARENT_SCOPE)

# ==== Macros to build the OSKAR apps module.
set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/oskar_apps.dir/src)
cuda_add_library(oskar_apps ${apps_SRC})

# Link in oskar_ms and oskar_fits if avaliable.
if(CASACORE_FOUND AND CFITSIO_FOUND)
    target_link_libraries(oskar_apps oskar oskar_widgets oskar_ms oskar_fits
        ${CASACORE_LIBRARIES} ${QT_QTCORE_LIBRARY} ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES})
elseif(CASACORE_FOUND)
    target_link_libraries(oskar_apps oskar oskar_widgets oskar_ms
        ${CASACORE_LIBRARIES} ${QT_QTCORE_LIBRARY} ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES})
elseif(CFITSIO_FOUND)
    target_link_libraries(oskar_apps oskar oskar_widgets oskar_fits
        ${CASACORE_LIBRARIES} ${QT_QTCORE_LIBRARY} ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES})
else()
    target_link_libraries(oskar_apps oskar oskar_widgets
        ${QT_QTCORE_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif()

set_target_properties(oskar_apps PROPERTIES
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
    INSTALL_RPATH_USE_LINK_PATH TRUE
    SOVERSION ${OSKAR_VERSION})

# Install target for the apps library.
install(TARGETS oskar_apps DESTINATION ${OSKAR_LIB_INSTALL_DIR})

# Tests
add_subdirectory(test)
