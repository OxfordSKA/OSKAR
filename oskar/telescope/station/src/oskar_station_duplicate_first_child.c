/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_duplicate_first_child(oskar_Station* station, int* status)
{
    int i = 0;
    if (*status || !station) return;

    /* Copy the first station to the others. */
    for (i = 1; i < station->num_elements; ++i)
    {
        oskar_station_free(oskar_station_child(station, i), status);
        station->child[i] = oskar_station_create_copy(
                oskar_station_child_const(station, 0), station->mem_location,
                status);
    }
}

#ifdef __cplusplus
}
#endif
