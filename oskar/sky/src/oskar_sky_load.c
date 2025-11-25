/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <stdio.h>
#include <stdlib.h>

#include "sky/oskar_sky.h"
#include "utility/oskar_getline.h"

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
        *status = OSKAR_ERR_BAD_DATA_TYPE;                /* LCOV_EXCL_LINE */
        return 0;                                         /* LCOV_EXCL_LINE */
    }

    /* Load from named columns if there's a format string in the text file. */
    sky = oskar_sky_load_named_columns(filename, type, status);
    if (*status == OSKAR_ERR_INVALID_ARGUMENT)
    {
        /* If invalid format string, return error to caller. */
        oskar_sky_free(sky, status);
        return 0;
    }
    if (!sky || *status == OSKAR_ERR_FILE_IO)
    {
        /* If error, clear status and proceed assuming original file format. */
        *status = 0;
        oskar_sky_free(sky, status);
    }
    else
    {
        /* No error, return a valid sky model. */
        return sky;
    }

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;                      /* LCOV_EXCL_LINE */
        return 0;                                         /* LCOV_EXCL_LINE */
    }

    /* Initialise the sky model. */
    sky = oskar_sky_create(type, OSKAR_CPU, 0, status);

    /* Loop over lines in file. */
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        int str_error = 0;

        /* Ensure enough space in arrays. */
        if (oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES) <= n)
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
    (void) fclose(file);

    /* Check if an error occurred. */
    if (*status)
    {
        oskar_sky_free(sky, status);                      /* LCOV_EXCL_LINE */
        sky = 0;                                          /* LCOV_EXCL_LINE */
    }

    /* Return a handle to the sky model. */
    return sky;
}

#ifdef __cplusplus
}
#endif
