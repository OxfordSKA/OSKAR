/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_load_layout(oskar_Station* station, int feed,
        const char* filename, int* status)
{
    /* Declare the line buffer and counter. */
    char* line = 0;
    size_t bufsize = 0;
    int n = 0, type = 0, old_size = 0;
    FILE* file = 0;
    if (*status || !station) return;

    /* Check type. */
    type = oskar_station_precision(station);
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Get the size of the station before loading the data. */
    old_size = oskar_station_num_elements(station);

    /* Loop over each line in the file. */
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        /* Declare parameter array. */
        /* x, y, z, delta x, delta y, delta z */
        double par[] = {0., 0., 0., 0., 0., 0.};
        const size_t num_par = sizeof(par) / sizeof(double);

        /* Load element data. */
        if (oskar_string_to_array_d(line, num_par, par) < 2) continue;

        /* Ensure the station model is big enough. */
        if (oskar_station_num_elements(station) <= n)
        {
            oskar_station_resize(station, n + 256, status);
            if (*status) break;
        }

        /* Get "true" coordinates ([3, 4, 5]) from "measured" coordinates. */
        par[3] += par[0];
        par[4] += par[1];
        par[5] += par[2];

        /* Store the data. */
        oskar_station_set_element_coords(station, feed, n,
                &par[0], &par[3], status);

        /* Increment element counter. */
        ++n;
    }

    /* Record number of elements loaded, if the station wasn't sized before. */
    if (old_size == 0)
    {
        oskar_station_resize(station, n, status);
    }
    else
    {
        /* Consistency check. */
        if (n != old_size)
        {
            *status = OSKAR_ERR_DIMENSION_MISMATCH;
        }
    }

    /* Free the line buffer and close the file. */
    free(line);
    fclose(file);
}

#ifdef __cplusplus
}
#endif
