/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/oskar_station.h"
#include "telescope/station/private_station.h"

#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"

#include <stdio.h>
#include <stdlib.h>
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_load_permitted_beams(oskar_Station* station,
        const char* filename, int* status)
{
    /* Declare the line buffer and counter. */
    char* line = 0;
    size_t bufsize = 0;
    int n = 0, type = 0;
    FILE* file = 0;
    oskar_Mem *az = 0, *el = 0;
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

    /* Get pointers to arrays to fill. */
    az = station->permitted_beam_az_rad;
    el = station->permitted_beam_el_rad;

    /* Loop over each line in the file. */
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        /* Declare parameter array. */
        /* azimuth (deg), elevation (deg) */
        double par[] = {0., 0.};
        size_t num_par = sizeof(par) / sizeof(double);

        /* Load element data. */
        if (oskar_string_to_array_d(line, num_par, par) < 2) continue;

        /* Ensure the arrays are big enough. */
        if (station->num_permitted_beams <= n)
        {
            oskar_mem_realloc(az, n + 100, status);
            oskar_mem_realloc(el, n + 100, status);
            station->num_permitted_beams = n + 100;
            if (*status) break;
        }

        /* Store the data. */
        oskar_mem_double(az, status)[n] = par[0] * M_PI / 180.0;
        oskar_mem_double(el, status)[n] = par[1] * M_PI / 180.0;

        /* Increment element counter. */
        ++n;
    }

    /* Set the final number of permitted beams. */
    station->num_permitted_beams = n;
    oskar_mem_realloc(az, n, status);
    oskar_mem_realloc(el, n, status);

    /* Free the line buffer and close the file. */
    if (line) free(line);
    fclose(file);
}

#ifdef __cplusplus
}
#endif
