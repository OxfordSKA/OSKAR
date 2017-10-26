#
# apps/CMakeLists.txt
#

add_subdirectory(test)

# === Include build macros used for apps.
include(${OSKAR_SOURCE_DIR}/cmake/oskar_build_macros.cmake)

if (CUDA_FOUND)
    oskar_app(
        NAME oskar_cuda_system_info SOURCES oskar_cuda_system_info_main.cpp)
endif()

macro(declare_oskar_apps)
    foreach (name ${ARGN})
        oskar_app(NAME ${name} SOURCES ${name}_main.cpp)
    endforeach()
endmacro()

declare_oskar_apps(
    oskar_convert_ecef_to_enu
    oskar_convert_geodetic_to_ecef
    oskar_binary_file_query
    oskar_filter_sky_model_clusters
    oskar_fit_element_data
    oskar_fits_image_to_sky_model
    oskar_imager
    oskar_sim_beam_pattern
    oskar_sim_interferometer
    oskar_vis_add
    oskar_vis_add_noise
    oskar_vis_summary
    oskar_vis_to_ms
    oskar_vis_upgrade_format
    oskar_vis_to_ascii_table)

set(IONOSPHERE_TESTING OFF)
if (IONOSPHERE_TESTING)
    # BINARY: oskar_sim_tec_screen
    set(name oskar_sim_tec_screen)
    oskar_app(NAME ${name} SOURCES ${name}_main.cpp NO_INSTALL)

    # BINARY: oskar_evaluate_pierce_points
    set(name oskar_evaluate_pierce_points)
    oskar_app(NAME ${name} SOURCES ${name}_main.cpp NO_INSTALL)

    #if (PNG_FOUND)
    #    include_directories(${PNG_INCLUDE_DIRS})
    #    set(name oskar_ionosphere_to_kml)
    #    oskar_app(NAME ${name} SOURCES ${name}_main.cpp
    #        EXTRA_LIBS ${PNG_LIBRARY}
    #        NO_INSTALL
    #    )
    #endif ()
endif()

# Install python scripts to the installation binary directory
file(GLOB app_scripts *.py)
foreach (script ${app_scripts})
    file(COPY ${script} DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endforeach()
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/"
    DESTINATION "${OSKAR_BIN_INSTALL_DIR}"
    FILES_MATCHING PATTERN "*.py"
    PERMISSIONS
        OWNER_READ OWNER_WRITE OWNER_EXECUTE
        GROUP_READ GROUP_EXECUTE
        WORLD_READ WORLD_EXECUTE
    PATTERN "CMakeFiles" EXCLUDE
    PATTERN "test" EXCLUDE
)
