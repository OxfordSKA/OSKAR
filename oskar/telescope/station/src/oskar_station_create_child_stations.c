/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_create_child_stations(oskar_Station* station,
        int* status)
{
    int i = 0, type = 0, location = 0;

    /* Check if safe to proceed. */
    if (*status || !station) return;

    /* Check that the memory isn't already allocated. */
    if (station->child)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return;
    }

    /* Allocate memory for child station array. */
    station->child = (oskar_Station**) calloc(
            station->num_elements, sizeof(oskar_Station*));
    if (!station->child)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return;
    }

    /* Create and initialise each child station. */
    type = oskar_station_precision(station);
    location = oskar_station_mem_location(station);
    for (i = 0; i < station->num_elements; ++i)
    {
        station->child[i] = oskar_station_create(type, location, 0, status);
    }
}

#ifdef __cplusplus
}
#endif
