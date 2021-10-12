/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_H_
#define OSKAR_TELESCOPE_H_

/**
 * @file oskar_telescope.h
 */

/* Public interface. */

#ifdef __cplusplus
extern "C" {
#endif

struct oskar_Telescope;
#ifndef OSKAR_TELESCOPE_TYPEDEF_
#define OSKAR_TELESCOPE_TYPEDEF_
typedef struct oskar_Telescope oskar_Telescope;
#endif /* OSKAR_TELESCOPE_TYPEDEF_ */

enum OSKAR_POL_MODE_TYPE
{
    OSKAR_POL_MODE_FULL,
    OSKAR_POL_MODE_SCALAR
};

#ifdef __cplusplus
}
#endif

#include <telescope/station/oskar_station.h>
#include <telescope/oskar_telescope_accessors.h>
#include <telescope/oskar_telescope_analyse.h>
#include <telescope/oskar_telescope_create.h>
#include <telescope/oskar_telescope_create_copy.h>
#include <telescope/oskar_telescope_free.h>
#include <telescope/oskar_telescope_load.h>
#include <telescope/oskar_telescope_load_pointing_file.h>
#include <telescope/oskar_telescope_load_position.h>
#include <telescope/oskar_telescope_load_station_coords_ecef.h>
#include <telescope/oskar_telescope_load_station_coords_enu.h>
#include <telescope/oskar_telescope_load_station_coords_wgs84.h>
#include <telescope/oskar_telescope_load_station_type_map.h>
#include <telescope/oskar_telescope_log_summary.h>
#include <telescope/oskar_telescope_override_element_cable_length_errors.h>
#include <telescope/oskar_telescope_override_element_gains.h>
#include <telescope/oskar_telescope_override_element_phases.h>
#include <telescope/oskar_telescope_resize.h>
#include <telescope/oskar_telescope_resize_station_array.h>
#include <telescope/oskar_telescope_save.h>
#include <telescope/oskar_telescope_save_layout.h>
#include <telescope/oskar_telescope_set_station_coords.h>
#include <telescope/oskar_telescope_set_station_coords_ecef.h>
#include <telescope/oskar_telescope_set_station_coords_enu.h>
#include <telescope/oskar_telescope_set_station_coords_wgs84.h>
#include <telescope/oskar_telescope_set_station_ids_and_coords.h>
#include <telescope/oskar_telescope_uvw.h>

#endif /* include guard */
