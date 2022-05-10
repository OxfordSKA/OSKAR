/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_telescope.h"
#include "telescope/oskar_telescope.h"

#include "mem/oskar_mem.h"

#include <stdlib.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Telescope* oskar_telescope_create(int type, int location,
        int num_stations, int* status)
{
    int i = 0;
    oskar_Telescope* telescope = 0;
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }
    telescope = (oskar_Telescope*) calloc(1, sizeof(oskar_Telescope));
    if (!telescope)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }
    telescope->precision = type;
    telescope->mem_location = location;
    telescope->pol_mode = OSKAR_POL_MODE_FULL;
    telescope->num_stations = num_stations;
    telescope->max_station_depth = 1;
    telescope->enable_numerical_patterns = 1;
    telescope->uv_filter_max = FLT_MAX;
    telescope->uv_filter_units = OSKAR_METRES;
    telescope->noise_seed = 1;
    for (i = 0; i < 3; ++i)
    {
        telescope->station_true_geodetic_rad[i] =
                oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, num_stations, status);
        telescope->station_true_offset_ecef_metres[i] =
                oskar_mem_create(type, location, num_stations, status);
        telescope->station_true_enu_metres[i] =
                oskar_mem_create(type, location, num_stations, status);
        telescope->station_measured_offset_ecef_metres[i] =
                oskar_mem_create(type, location, num_stations, status);
        telescope->station_measured_enu_metres[i] =
                oskar_mem_create(type, location, num_stations, status);
    }
    telescope->station_type_map =
            oskar_mem_create(OSKAR_INT, OSKAR_CPU, num_stations, status);
    oskar_mem_clear_contents(telescope->station_type_map, status);
    telescope->tec_screen_path =
            oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
    telescope->gains = oskar_gains_create(type);
    telescope->harp_freq_cpu = oskar_mem_create(
            OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    return telescope;
}

#ifdef __cplusplus
}
#endif
