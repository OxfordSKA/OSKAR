#
# oskar/telescope/CMakeLists.txt
#

set(telescope_SRC
    src/oskar_telescope_accessors.c
    src/oskar_telescope_analyse.c
    src/oskar_telescope_create.c
    src/oskar_telescope_create_copy.c
    src/oskar_telescope_free.c
    src/oskar_telescope_load.cpp
    src/oskar_telescope_load_pointing_file.c
    src/oskar_telescope_load_position.c
    src/oskar_telescope_load_station_coords_ecef.c
    src/oskar_telescope_load_station_coords_enu.c
    src/oskar_telescope_load_station_coords_wgs84.c
    src/oskar_telescope_load_station_type_map.c
    src/oskar_telescope_log_summary.c
    src/oskar_telescope_override_element_cable_length_errors.c
    src/oskar_telescope_override_element_gains.c
    src/oskar_telescope_override_element_phases.c
    src/oskar_telescope_resize.c
    src/oskar_telescope_resize_station_array.c
    src/oskar_telescope_save.c
    src/oskar_telescope_save_layout.c
    src/oskar_telescope_set_station_coords.c
    src/oskar_telescope_set_station_coords_ecef.c
    src/oskar_telescope_set_station_coords_enu.c
    src/oskar_telescope_set_station_coords_wgs84.c
    src/oskar_telescope_set_station_ids_and_coords.c
    src/oskar_telescope_uvw.c
    src/oskar_TelescopeLoadAbstract.cpp
    src/private_TelescopeLoaderApodisation.cpp
    src/private_TelescopeLoaderCableLengthError.cpp
    src/private_TelescopeLoaderElementPattern.cpp
    src/private_TelescopeLoaderElementTypes.cpp
    src/private_TelescopeLoaderFeedAngle.cpp
    src/private_TelescopeLoaderGainModel.cpp
    src/private_TelescopeLoaderGainPhase.cpp
    src/private_TelescopeLoaderHarpData.cpp
    src/private_TelescopeLoaderLayout.cpp
    src/private_TelescopeLoaderMountTypes.cpp
    src/private_TelescopeLoaderNoise.cpp
    src/private_TelescopeLoaderPermittedBeams.cpp
    src/private_TelescopeLoaderPosition.cpp
    src/private_TelescopeLoaderStationTypeMap.cpp
)

# Add contents of station subdirectory.
add_subdirectory(station)
foreach (file ${station_SRC})
    list(APPEND telescope_SRC station/${file})
endforeach()

set(telescope_SRC "${telescope_SRC}" PARENT_SCOPE)

if (BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
    add_subdirectory(test)
endif()
