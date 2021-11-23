/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/oskar_telescope.h"
#include "telescope/private_telescope.h"
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_load_position(oskar_Telescope* telescope,
        const char* filename, int* status)
{
    int num_coords = 0;
    oskar_Mem *lon = 0, *lat = 0, *alt = 0;

    /* Load columns from file. */
    lon = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    lat = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    alt = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, status);
    num_coords = (int) oskar_mem_load_ascii(filename, 3, status,
            lon, "", lat, "", alt, "0.0");

    /* Set the telescope centre coordinates. */
    if (num_coords == 1)
    {
        telescope->lon_rad = (oskar_mem_double(lon, status))[0] * M_PI / 180.0;
        telescope->lat_rad = (oskar_mem_double(lat, status))[0] * M_PI / 180.0;
        telescope->alt_metres = (oskar_mem_double(alt, status))[0];
    }
    else
    {
        *status = OSKAR_ERR_BAD_COORD_FILE;
    }

    /* Free memory. */
    oskar_mem_free(lon, status);
    oskar_mem_free(lat, status);
    oskar_mem_free(alt, status);
}

#ifdef __cplusplus
}
#endif
