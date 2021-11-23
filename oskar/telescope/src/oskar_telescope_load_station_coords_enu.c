/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/oskar_telescope.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_load_station_coords_enu(oskar_Telescope* telescope,
        const char* filename, double longitude, double latitude,
        double altitude, int* status)
{
    int num_stations = 0;
    oskar_Mem *x = 0, *y = 0, *z = 0, *x_err = 0, *y_err = 0, *z_err = 0;

    /* Load columns from file into memory. */
    x     = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    y     = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    z     = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    x_err = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    y_err = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    z_err = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    num_stations = (int) oskar_mem_load_ascii(filename, 6, status,
            x, "", y, "", z, "0.0", x_err, "0.0", y_err, "0.0", z_err, "0.0");

    /* Set the station coordinates. */
    oskar_telescope_set_station_coords_enu(telescope, longitude, latitude,
            altitude, num_stations, x, y, z, x_err, y_err, z_err, status);

    /* Free memory. */
    oskar_mem_free(x, status);
    oskar_mem_free(y, status);
    oskar_mem_free(z, status);
    oskar_mem_free(x_err, status);
    oskar_mem_free(y_err, status);
    oskar_mem_free(z_err, status);
}

#ifdef __cplusplus
}
#endif
