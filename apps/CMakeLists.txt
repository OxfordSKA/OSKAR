#
# src/apps/CMakeLists.txt
#

# Recurse into lib subdirectory.
add_subdirectory(lib)

# BINARY: oskar_sim
include_directories(${QT_INCLUDE_DIR})
add_executable(oskar_sim oskar_sim.cpp)
target_link_libraries(oskar_sim oskar oskar_apps ${QT_QTCORE_LIBRARY})
set_target_properties(oskar_sim PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS oskar_sim DESTINATION ${OSKAR_BIN_INSTALL_DIR})

# BINARY: oskar_sim_scalar
include_directories(${QT_INCLUDE_DIR})
add_executable(oskar_sim_scalar oskar_sim_scalar.cpp)
target_link_libraries(oskar_sim_scalar oskar oskar_apps ${QT_QTCORE_LIBRARY})
set_target_properties(oskar_sim_scalar PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS oskar_sim_scalar DESTINATION ${OSKAR_BIN_INSTALL_DIR})

# BINARY: oskar_sim_benchmark
include_directories(${QT_INCLUDE_DIR})
add_executable(oskar_sim_benchmark oskar_sim_benchmark.cpp)
target_link_libraries(oskar_sim_benchmark oskar oskar_apps ${QT_QTCORE_LIBRARY})
set_target_properties(oskar_sim_benchmark PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
set_target_properties(oskar_sim_benchmark PROPERTIES 
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}" LINK_FLAGS "${OpenMP_CXX_FLAGS}")
install(TARGETS oskar_sim_benchmark DESTINATION ${OSKAR_BIN_INSTALL_DIR})

# BINARY: oskar_beam_pattern
include_directories(${QT_INCLUDE_DIR})
add_executable(oskar_beam_pattern oskar_beam_pattern.cpp)
target_link_libraries(oskar_beam_pattern oskar oskar_apps ${QT_QTCORE_LIBRARY})
set_target_properties(oskar_beam_pattern PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS oskar_beam_pattern DESTINATION ${OSKAR_BIN_INSTALL_DIR})

# BINARY: oskar_icrs_to_az_el
add_executable(oskar_icrs_to_az_el oskar_icrs_to_az_el.cpp)
target_link_libraries(oskar_icrs_to_az_el oskar)
set_target_properties(oskar_icrs_to_az_el PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

if (BUILD_OSKAR_WIDGETS)
    include_directories(${Qwt5_INCLUDE_DIR})
    include_directories(${QT_QTGUI_INCLUDE_DIR})
    include_directories(${QT_INCLUDE_DIR})
    include_directories(${QT_QTCORE_INCLUDE_DIR})

    # BINARY: oskar_viewer_sclar
    add_executable(oskar_viewer_scalar oskar_viewer_scalar.cpp)
    target_link_libraries(oskar_viewer_scalar oskar oskar_apps oskar_widgets
        ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY})
    set_target_properties(oskar_viewer_scalar PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
    #install(TARGETS oskar_viewer_scalar DESTINATION ${OSKAR_BIN_INSTALL_DIR})

endif ()

cuda_add_executable(oskar_device_info device_info.cpp)

