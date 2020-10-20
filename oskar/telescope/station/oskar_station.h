/*
 * Copyright (c) 2012-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_STATION_H_
#define OSKAR_STATION_H_

/**
 * @file oskar_station.h
 */

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Station;
#ifndef OSKAR_STATION_TYPEDEF_
#define OSKAR_STATION_TYPEDEF_
typedef struct oskar_Station oskar_Station;
#endif /* OSKAR_STATION_TYPEDEF_ */

enum OSKAR_STATION_TYPE
{
    OSKAR_STATION_TYPE_AA,
    OSKAR_STATION_TYPE_ISOTROPIC,
    OSKAR_STATION_TYPE_GAUSSIAN_BEAM,
    OSKAR_STATION_TYPE_VLA_PBCOR
};

#ifdef __cplusplus
}
#endif

#include <telescope/station/element/oskar_element.h>
#include <telescope/station/oskar_station_work.h>
#include <telescope/station/oskar_station_accessors.h>
#include <telescope/station/oskar_station_analyse.h>
#include <telescope/station/oskar_station_beam.h>
#include <telescope/station/oskar_station_beam_horizon_direction.h>
#include <telescope/station/oskar_station_create_child_stations.h>
#include <telescope/station/oskar_station_create_copy.h>
#include <telescope/station/oskar_station_create.h>
#include <telescope/station/oskar_station_different.h>
#include <telescope/station/oskar_station_duplicate_first_child.h>
#include <telescope/station/oskar_station_free.h>
#include <telescope/station/oskar_station_load_apodisation.h>
#include <telescope/station/oskar_station_load_cable_length_error.h>
#include <telescope/station/oskar_station_load_element_types.h>
#include <telescope/station/oskar_station_load_feed_angle.h>
#include <telescope/station/oskar_station_load_gain_phase.h>
#include <telescope/station/oskar_station_load_layout.h>
#include <telescope/station/oskar_station_load_mount_types.h>
#include <telescope/station/oskar_station_load_permitted_beams.h>
#include <telescope/station/oskar_station_override_element_cable_length_errors.h>
#include <telescope/station/oskar_station_override_element_feed_angle.h>
#include <telescope/station/oskar_station_override_element_gains.h>
#include <telescope/station/oskar_station_override_element_phases.h>
#include <telescope/station/oskar_station_override_element_time_variable_gains.h>
#include <telescope/station/oskar_station_override_element_time_variable_phases.h>
#include <telescope/station/oskar_station_override_element_xy_position_errors.h>
#include <telescope/station/oskar_station_resize.h>
#include <telescope/station/oskar_station_resize_element_types.h>
#include <telescope/station/oskar_station_save_apodisation.h>
#include <telescope/station/oskar_station_save_cable_length_error.h>
#include <telescope/station/oskar_station_save_element_types.h>
#include <telescope/station/oskar_station_save_feed_angle.h>
#include <telescope/station/oskar_station_save_gain_phase.h>
#include <telescope/station/oskar_station_save_layout.h>
#include <telescope/station/oskar_station_save_mount_types.h>
#include <telescope/station/oskar_station_save_permitted_beams.h>
#include <telescope/station/oskar_station_set_element_cable_length_error.h>
#include <telescope/station/oskar_station_set_element_coords.h>
#include <telescope/station/oskar_station_set_element_errors.h>
#include <telescope/station/oskar_station_set_element_feed_angle.h>
#include <telescope/station/oskar_station_set_element_mount_type.h>
#include <telescope/station/oskar_station_set_element_type.h>
#include <telescope/station/oskar_station_set_element_weight.h>

#endif /* include guard */
