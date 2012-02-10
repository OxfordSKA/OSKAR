#
# src/apps/CMakeLists.txt
#

# Recurse into lib subdirectory.
add_subdirectory(lib)

include_directories(${QT_INCLUDE_DIR})

# BINARY: oskar_sim
set(bin_name oskar_sim)
add_executable(${bin_name} ${bin_name}_main.cpp)
target_link_libraries(${bin_name} oskar oskar_apps ${QT_QTCORE_LIBRARY})
set_target_properties(${bin_name} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
set_target_properties(${bin_name} PROPERTIES
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}" LINK_FLAGS "${OpenMP_CXX_FLAGS}")
install(TARGETS ${bin_name} DESTINATION ${OSKAR_BIN_INSTALL_DIR})

# BINARY: oskar_sim_scalar
add_executable(oskar_sim_scalar oskar_sim_scalar.cpp)
target_link_libraries(oskar_sim_scalar oskar oskar_apps ${QT_QTCORE_LIBRARY})
set_target_properties(oskar_sim_scalar PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS oskar_sim_scalar DESTINATION ${OSKAR_BIN_INSTALL_DIR})

# BINARY: oskar_beam_pattern
add_executable(oskar_beam_pattern oskar_beam_pattern.cpp)
target_link_libraries(oskar_beam_pattern oskar oskar_apps ${QT_QTCORE_LIBRARY})
set_target_properties(oskar_beam_pattern PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
install(TARGETS oskar_beam_pattern DESTINATION ${OSKAR_BIN_INSTALL_DIR})

# BINARY: oskar_icrs_to_az_el
add_executable(oskar_icrs_to_az_el oskar_icrs_to_az_el.cpp)
target_link_libraries(oskar_icrs_to_az_el oskar)
set_target_properties(oskar_icrs_to_az_el PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)

# BINARY: oskar_device_info
cuda_add_executable(oskar_device_info device_info.cpp)
target_link_libraries(oskar_device_info ${QT_QTCORE_LIBRARY})

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

    # BINARY: oskar_sim_gui
    set(bin_name oskar_sim_gui)
    add_executable(${bin_name} ${bin_name}_main.cpp)
    target_link_libraries(${bin_name} 
        oskar oskar_apps ${QT_QTCORE_LIBRARY} ${QT_QTGUI_LIBRARY})
    set_target_properties(${bin_name} PROPERTIES INSTALL_RPATH_USE_LINK_PATH TRUE)
    set_target_properties(${bin_name} PROPERTIES
        COMPILE_FLAGS "${OpenMP_CXX_FLAGS}"
        LINK_FLAGS "${OpenMP_CXX_FLAGS}")
    install(TARGETS ${bin_name} DESTINATION ${OSKAR_BIN_INSTALL_DIR})
endif ()
