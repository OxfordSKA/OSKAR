/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_telescope.h"
#include "telescope/oskar_telescope.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Telescope* oskar_telescope_create_copy(const oskar_Telescope* src,
        int location, int* status)
{
    int i = 0;
    oskar_Telescope* telescope = 0;

    /* Create a new, empty model. */
    telescope = oskar_telescope_create(oskar_telescope_precision(src),
            location, 0, status);

    /* Copy private meta-data. */
    telescope->precision = src->precision;
    telescope->mem_location = location;

    /* Copy the meta-data. */
    telescope->pol_mode = src->pol_mode;
    telescope->num_stations = src->num_stations;
    telescope->num_station_models = src->num_station_models;
    telescope->max_station_size = src->max_station_size;
    telescope->max_station_depth = src->max_station_depth;
    telescope->allow_station_beam_duplication = src->allow_station_beam_duplication;
    telescope->enable_numerical_patterns = src->enable_numerical_patterns;
    telescope->lon_rad = src->lon_rad;
    telescope->lat_rad = src->lat_rad;
    telescope->alt_metres = src->alt_metres;
    telescope->pm_x_rad = src->pm_x_rad;
    telescope->pm_y_rad = src->pm_y_rad;
    telescope->phase_centre_coord_type = src->phase_centre_coord_type;
    telescope->phase_centre_rad[0] = src->phase_centre_rad[0];
    telescope->phase_centre_rad[1] = src->phase_centre_rad[1];
    telescope->channel_bandwidth_hz = src->channel_bandwidth_hz;
    telescope->time_average_sec = src->time_average_sec;
    telescope->uv_filter_min = src->uv_filter_min;
    telescope->uv_filter_max = src->uv_filter_max;
    telescope->uv_filter_units = src->uv_filter_units;
    telescope->noise_enabled = src->noise_enabled;
    telescope->noise_seed = src->noise_seed;
    telescope->ionosphere_screen_type = src->ionosphere_screen_type;
    telescope->tec_screen_height_km = src->tec_screen_height_km;
    telescope->tec_screen_pixel_size_m = src->tec_screen_pixel_size_m;
    telescope->tec_screen_time_interval_sec = src->tec_screen_time_interval_sec;
    telescope->isoplanatic_screen = src->isoplanatic_screen;

    /* Copy the coordinates. */
    for (i = 0; i < 3; ++i)
    {
        oskar_mem_copy(telescope->station_true_geodetic_rad[i],
                src->station_true_geodetic_rad[i], status);
        oskar_mem_copy(telescope->station_true_offset_ecef_metres[i],
                src->station_true_offset_ecef_metres[i], status);
        oskar_mem_copy(telescope->station_true_enu_metres[i],
                src->station_true_enu_metres[i], status);
        oskar_mem_copy(telescope->station_measured_offset_ecef_metres[i],
                src->station_measured_offset_ecef_metres[i], status);
        oskar_mem_copy(telescope->station_measured_enu_metres[i],
                src->station_measured_enu_metres[i], status);
    }
    oskar_mem_copy(telescope->station_type_map, src->station_type_map, status);
    oskar_mem_copy(telescope->tec_screen_path, src->tec_screen_path, status);

    /* Copy the gain model. */
    oskar_gains_free(telescope->gains, status);
    telescope->gains = oskar_gains_create_copy(src->gains, status);

    /* Copy the HARP data. */
    telescope->harp_num_freq = src->harp_num_freq;
    oskar_mem_copy(telescope->harp_freq_cpu, src->harp_freq_cpu, status);
    if (src->harp_num_freq > 0)
    {
        telescope->harp_data = (oskar_Harp**) calloc(
                src->harp_num_freq, sizeof(oskar_Harp*));
        for (i = 0; i < src->harp_num_freq; ++i)
        {
            telescope->harp_data[i] = oskar_harp_create_copy(
                    src->harp_data[i], status);
        }
    }

    /* Copy each station. */
    telescope->station = (oskar_Station**) calloc(
            src->num_station_models, sizeof(oskar_Station*));
    for (i = 0; i < src->num_station_models; ++i)
    {
        telescope->station[i] = oskar_station_create_copy(
                oskar_telescope_station_const(src, i), location, status);
    }

    /* Return pointer to data structure. */
    return telescope;
}

#ifdef __cplusplus
}
#endif
