/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/oskar_telescope.h"

#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_load_cable_length_error(
        oskar_Telescope* telescope,
        int feed,
        const char* filename,
        int* status
)
{
    char* line = 0;
    size_t bufsize = 0;
    int n = 0;
    FILE* file = 0;
    if (*status || !telescope) return;
    const int type = oskar_telescope_precision(telescope);
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
    const int old_size = oskar_telescope_num_stations(telescope);
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        double par[] = {0.0};
        size_t num_par = sizeof(par) / sizeof(double);
        if (oskar_string_to_array_d(line, num_par, par) < 1) continue;
        oskar_telescope_set_station_cable_length_error(
                telescope, feed, n, par[0], status
        );
        ++n;
    }

    /* Consistency check with previous telescope size (should be the same as
     * the number of elements loaded). */
    if (!*status && n != old_size)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
    }
    free(line);
    (void) fclose(file);
}

#ifdef __cplusplus
}
#endif
