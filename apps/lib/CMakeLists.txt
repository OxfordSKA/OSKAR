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
    src/oskar_settings_free.c
    src/oskar_settings_load_image.cpp
    src/oskar_settings_load_observation.cpp
    src/oskar_settings_load_simulator.cpp
    src/oskar_settings_load_sky.cpp
    src/oskar_settings_load_telescope.cpp
    src/oskar_settings_load.c
    src/oskar_settings_print.c
    src/oskar_sim_beam_pattern.cpp
    src/oskar_sim_interferometer.cpp
)

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
if(BUILD_OSKAR_MS)
    target_link_libraries(oskar_apps oskar oskar_ms ${CASACORE_LIBRARIES}
        ${QT_QTCORE_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
else()
    target_link_libraries(oskar_apps oskar
        ${QT_QTCORE_LIBRARY} ${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES})
endif()

set_target_properties(oskar_apps
    PROPERTIES
    INSTALL_RPATH ${CMAKE_INSTALL_PREFIX}/${OSKAR_LIB_INSTALL_DIR}
    INSTALL_RPATH_USE_LINK_PATH TRUE
    COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
    LINK_FLAGS "${OpenMP_CXX_FLAGS}"
    SOVERSION ${oskar-lib_SOVERSION})

# Install target for the apps library.
install(TARGETS oskar_apps DESTINATION ${OSKAR_LIB_INSTALL_DIR})

# Build MATLAB mex functions.
add_subdirectory(matlab)

# Tests
add_subdirectory(test)
