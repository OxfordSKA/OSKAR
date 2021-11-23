/*
 * Copyright (c) 2021, The OSKAR Developers.
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

void oskar_telescope_resize_station_array(oskar_Telescope* telescope,
        int size, int* status)
{
    int i = 0;
    if (*status) return;

    /* Get the old size. */
    const int old_size = telescope->num_station_models;

    /* Check if increasing or decreasing in size. */
    if (size > old_size)
    {
        /* Enlarge the station array and create new stations. */
        telescope->station = (oskar_Station**) realloc(telescope->station,
                size * sizeof(oskar_Station*));
        if (!telescope->station)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            return;
        }
        for (i = old_size; i < size; ++i)
        {
            telescope->station[i] = oskar_station_create(
                    oskar_mem_type(telescope->station_true_offset_ecef_metres[0]),
                    oskar_mem_location(telescope->station_true_offset_ecef_metres[0]),
                    0, status);
        }
    }
    else if (size < old_size)
    {
        /* Free old stations and shrink the station array. */
        for (i = size; i < old_size; ++i)
        {
            oskar_station_free(oskar_telescope_station(telescope, i), status);
        }
        telescope->station = (oskar_Station**) realloc(telescope->station,
                size * sizeof(oskar_Station*));
    }
    else
    {
        /* No resize necessary. */
        return;
    }

    /* Store the new size. */
    telescope->num_station_models = size;
}

#ifdef __cplusplus
}
#endif
