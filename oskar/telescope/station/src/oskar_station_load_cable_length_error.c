/*
 * Copyright (c) 2019-2021, The OSKAR Developers.
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

void oskar_station_load_cable_length_error(oskar_Station* station, int feed,
        const char* filename, int* status)
{
    char* line = 0;
    size_t bufsize = 0;
    int n = 0;
    FILE* file = 0;
    if (*status || !station) return;
    const int type = oskar_station_precision(station);
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    const int old_size = oskar_station_num_elements(station);
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        double par[] = {0.0};
        size_t num_par = sizeof(par) / sizeof(double);
        if (oskar_string_to_array_d(line, num_par, par) < 1) continue;
        if (oskar_station_num_elements(station) <= n)
        {
            oskar_station_resize(station, n + 256, status);
            if (*status) break;
        }
        oskar_station_set_element_cable_length_error(station, feed,
                n, par[0], status);
        ++n;
    }

    /* Consistency check with previous station size (should be the same as
     * the number of elements loaded). */
    if (!*status && n != old_size)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
    }
    free(line);
    fclose(file);
}

#ifdef __cplusplus
}
#endif
