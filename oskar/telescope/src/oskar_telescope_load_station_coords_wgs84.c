/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/oskar_telescope.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_load_station_coords_wgs84(oskar_Telescope* telescope,
        const char* filename, double longitude, double latitude,
        double altitude, int* status)
{
    int num_stations = 0;
    oskar_Mem *lon_deg = 0, *lat_deg = 0, *alt_m = 0;

    /* Load columns from file into memory. */
    lon_deg = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    lat_deg = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    alt_m   = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    num_stations = (int) oskar_mem_load_ascii(filename, 3, status,
            lon_deg, "", lat_deg, "", alt_m, "0.0");

    /* Set the station coordinates. */
    oskar_telescope_set_station_coords_wgs84(telescope, longitude, latitude,
            altitude, num_stations, lon_deg, lat_deg, alt_m, status);

    /* Free memory. */
    oskar_mem_free(lon_deg, status);
    oskar_mem_free(lat_deg, status);
    oskar_mem_free(alt_m, status);
}

#ifdef __cplusplus
}
#endif
