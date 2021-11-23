/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"
#include "utility/oskar_getline.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Sky* oskar_sky_load(const char* filename, int type, int* status)
{
    int n = 0;
    FILE* file = 0;
    char* line = 0;
    size_t bufsize = 0;
    oskar_Sky* sky = 0;
    if (*status) return 0;

    /* Get the data type. */
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return 0;
    }

    /* Initialise the sky model. */
    sky = oskar_sky_create(type, OSKAR_CPU, 0, status);

    /* Loop over lines in file. */
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        int str_error = 0;

        /* Ensure enough space in arrays. */
        if (oskar_sky_num_sources(sky) <= n)
        {
            const int new_size = ((2 * n) < 100) ? 100 : (2 * n);
            oskar_sky_resize(sky, new_size, status);
            if (*status) break;
        }

        /* Try to set source data from the string. */
        oskar_sky_set_source_str(sky, n, line, &str_error);

        /* Increment source count only if successful. */
        if (!str_error) n++;
    }

    /* Set the size to be the actual number of elements loaded. */
    oskar_sky_resize(sky, n, status);

    /* Free the line buffer and close the file. */
    free(line);
    fclose(file);

    /* Check if an error occurred. */
    if (*status)
    {
        oskar_sky_free(sky, status);
        sky = 0;
    }

    /* Return a handle to the sky model. */
    return sky;
}

#ifdef __cplusplus
}
#endif
