#
# src/apps/lib/CMakeLists.txt
#

set(apps_SRC
    src/oskar_file_utils.cpp
    src/oskar_imager_dft.cu
    src/oskar_load_stations.cpp
    src/oskar_set_up_sky.cpp
    src/oskar_set_up_telescope.cpp
    src/oskar_set_up_visibilities.cpp
    src/oskar_Settings.cpp
    src/oskar_SettingsBenchmark.cpp
    src/oskar_SettingsImage.cpp
    src/oskar_SettingsObservation.cpp
    src/oskar_SettingsSky.cpp
    src/oskar_sim.cpp
)

if (BUILD_OSKAR_WIDGETS)
    include_directories(${QT_QTGUI_INCLUDE_DIR})
    include_directories(${QT_INCLUDE_DIR})
    include_directories(${QT_QTCORE_INCLUDE_DIR})
    list(APPEND apps_SRC
        src/oskar_MainWindow.cpp
    )

    set(apps_MOC_SRC
        oskar_MainWindow.h
    )
    qt4_wrap_cpp(apps_SRC ${apps_MOC_SRC})
endif()

if(BUILD_OSKAR_MS)
    include_directories(${CASACORE_INCLUDE_DIR}/casacore)
    list(APPEND apps_SRC src/oskar_write_ms.cpp)
endif()
set(apps_SRC "${apps_SRC}" PARENT_SCOPE)

# ==== Macros to build the oskar-lib apps module.
include_directories(${QT_INCLUDE_DIR})
include_directories(${CUDA_INCLUDE_DIRS})
set(CUDA_GENERATED_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/oskar_apps.dir/src)
cuda_add_library(oskar_apps ${apps_SRC})

# Link in oskar_ms if its avaliable
## FIXME: mmm we might want to rethink the dependencies of the apps library
## to not link vs other oskar libaries.
## ... if this is possible ...
if (BUILD_OSKAR_MS AND BUILD_OSKAR_WIDGETS)
    target_link_libraries(oskar_apps oskar oskar_widgets oskar_ms
        ${CASACORE_LIBRARIES}
        ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY}
        ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
elseif(BUILD_OSKAR_MS)
    target_link_libraries(oskar_apps oskar oskar_ms ${CASACORE_LIBRARIES}
        ${QT_QTCORE_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
else()
    target_link_libraries(oskar_apps oskar
        ${QT_QTCORE_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif()

set_target_properties(oskar_apps 
    PROPERTIES COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}")

set_target_properties(oskar_apps PROPERTIES LINKER_LANGUAGE CXX)

set_target_properties(oskar_apps PROPERTIES
    INSTALL_RPATH_USE_LINK_PATH TRUE
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/lib)

# Install target for the apps library.
set_target_properties(oskar_apps PROPERTIES SOVERSION ${oskar-lib_SOVERSION})
install(TARGETS oskar_apps DESTINATION ${OSKAR_LIB_INSTALL_DIR})

# Build MATLAB mex functions.
add_subdirectory(matlab)

# Tests
add_subdirectory(test)
