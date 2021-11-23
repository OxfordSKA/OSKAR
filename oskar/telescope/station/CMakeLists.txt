#
# oskar/telescope/station/CMakeLists.txt
#

set(station_SRC
    define_blank_below_horizon.h
    define_evaluate_element_weights_dft.h
    define_evaluate_element_weights_errors.h
    define_evaluate_tec_screen.h
    define_evaluate_vla_beam_pbcor.h
    src/oskar_blank_below_horizon.c
    #src/oskar_evaluate_pierce_points.c
    src/oskar_evaluate_element_weights_dft.c
    src/oskar_evaluate_element_weights_errors.c
    src/oskar_evaluate_tec_screen.c
    src/oskar_evaluate_station_beam_aperture_array.c
    src/oskar_evaluate_station_beam_gaussian.c
    #src/oskar_evaluate_station_from_telescope_dipole_azimuth.c
    src/oskar_evaluate_vla_beam_pbcor.c
    src/oskar_station_accessors.c
    src/oskar_station_analyse.c
    src/oskar_station_beam.c
    src/oskar_station_beam_horizon_direction.c
    src/oskar_station_create_child_stations.c
    src/oskar_station_create_copy.c
    src/oskar_station_create.c
    src/oskar_station_different.c
    src/oskar_station_duplicate_first_child.c
    src/oskar_station_evaluate_element_weights.c
    src/oskar_station_free.c
    src/oskar_station_load_apodisation.c
    src/oskar_station_load_cable_length_error.c
    src/oskar_station_load_element_types.c
    src/oskar_station_load_feed_angle.c
    src/oskar_station_load_gain_phase.c
    src/oskar_station_load_layout.c
    src/oskar_station_load_mount_types.c
    src/oskar_station_load_permitted_beams.c
    src/oskar_station_override_element_cable_length_errors.c
    src/oskar_station_override_element_feed_angle.c
    src/oskar_station_override_element_gains.c
    src/oskar_station_override_element_phases.c
    src/oskar_station_override_element_time_variable_gains.c
    src/oskar_station_override_element_time_variable_phases.c
    src/oskar_station_override_element_xy_position_errors.c
    src/oskar_station_resize_element_types.c
    src/oskar_station_resize.c
    src/oskar_station_save_apodisation.c
    src/oskar_station_save_cable_length_error.c
    src/oskar_station_save_element_types.c
    src/oskar_station_save_feed_angle.c
    src/oskar_station_save_gain_phase.c
    src/oskar_station_save_layout.c
    src/oskar_station_save_mount_types.c
    src/oskar_station_save_permitted_beams.c
    src/oskar_station_set_element_coords.c
    src/oskar_station_set_element_cable_length_error.c
    src/oskar_station_set_element_feed_angle.c
    src/oskar_station_set_element_errors.c
    src/oskar_station_set_element_mount_type.c
    src/oskar_station_set_element_type.c
    src/oskar_station_set_element_weight.c
    src/oskar_station_work.c
    src/oskar_station.cl
)

if (CUDA_FOUND)
    list(APPEND station_SRC src/oskar_station.cu)
endif()

# Add contents of station element subdirectory.
add_subdirectory(element)
foreach (file ${element_SRC})
    list(APPEND station_SRC element/${file})
endforeach()

set(station_SRC "${station_SRC}" PARENT_SCOPE)

# ==== Recurse into test subdirectory.
if (BUILD_TESTING OR NOT DEFINED BUILD_TESTING)
    add_subdirectory(test)
endif()
