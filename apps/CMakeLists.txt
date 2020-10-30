#
# apps/CMakeLists.txt
#

add_subdirectory(test)

# === Include build macros used for apps.
include(${OSKAR_SOURCE_DIR}/cmake/oskar_build_macros.cmake)

if (CASACORE_FOUND)
    oskar_app(
        NAME oskar_vis_to_ms SOURCES oskar_vis_to_ms_main.cpp)
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
    oskar_system_info
    oskar_vis_add
    oskar_vis_add_noise
    oskar_vis_summary)

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
