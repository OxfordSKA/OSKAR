/*
 * Copyright (c) 2015-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/oskar_station.h"

#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_load_feed_angle(oskar_Station* station, int feed,
        const char* filename, int* status)
{
    char* line = 0;
    size_t bufsize = 0;
    int i = 0, n = 0;
    FILE* file = 0;
    double par[] = {0.0, 0.0, 0.0};
    const size_t num_par = sizeof(par) / sizeof(double);
    if (*status || !station) return;

    /* Check type. */
    const int type = oskar_station_precision(station);
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
    const int old_size = oskar_station_num_elements(station);

    /* Loop over each line in the file. */
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        /* Load element data and store it. */
        par[0] = par[1] = par[2] = 0.0;
        if (oskar_string_to_array_d(line, num_par, par) < 1) continue;
        oskar_station_set_element_feed_angle(
                station, feed, n, par[0], par[1], par[2], status
        );

        /* Increment element counter. */
        ++n;
    }

    /* Consistency check. */
    if (!*status)
    {
        /* If we have data for only a single element, copy it to all others. */
        if (n == 1)
        {
            for (i = 1; i < old_size; ++i)
            {
                oskar_station_set_element_feed_angle(
                        station, feed, i, par[0], par[1], par[2], status
                );
            }
        }
        else if (n != old_size)
        {
            *status = OSKAR_ERR_DIMENSION_MISMATCH;
        }
    }

    /* Free the line buffer and close the file. */
    free(line);
    (void) fclose(file);
}

#ifdef __cplusplus
}
#endif
