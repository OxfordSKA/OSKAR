/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
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

void oskar_station_load_element_types(oskar_Station* station,
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
        int par = 0;

        /* Load element data. */
        if (oskar_string_to_array_i(line, 1, &par) < 1) continue;

        /* Ensure the station model is big enough. */
        if (oskar_station_num_elements(station) <= n)
        {
            oskar_station_resize(station, n + 256, status);
            if (*status) break;
        }

        /* Store the data. */
        oskar_station_set_element_type(station, n, par, status);

        /* Increment element counter. */
        ++n;
    }

    /* Consistency check with previous station size (should be the same as
     * the number of elements loaded). */
    if (!*status && n != old_size)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
    }

    /* Free the line buffer and close the file. */
    if (line) free(line);
    fclose(file);
}

#ifdef __cplusplus
}
#endif
