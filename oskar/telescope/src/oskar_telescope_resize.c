/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/private_telescope.h"
#include "telescope/oskar_telescope.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_resize(oskar_Telescope* telescope, int size, int* status)
{
    int i = 0;
    if (*status) return;

    /* Do not resize the station array here. */
    oskar_mem_realloc(telescope->station_type_map, size, status);

    /* Resize the coordinate arrays. */
    for (i = 0; i < 3; ++i)
    {
        oskar_mem_realloc(telescope->station_true_geodetic_rad[i],
                size, status);
        oskar_mem_realloc(telescope->station_true_offset_ecef_metres[i],
                size, status);
        oskar_mem_realloc(telescope->station_true_enu_metres[i],
                size, status);
        oskar_mem_realloc(telescope->station_measured_offset_ecef_metres[i],
                size, status);
        oskar_mem_realloc(telescope->station_measured_enu_metres[i],
                size, status);
    }

    /* Store the new size. */
    telescope->num_stations = size;
}

#ifdef __cplusplus
}
#endif
