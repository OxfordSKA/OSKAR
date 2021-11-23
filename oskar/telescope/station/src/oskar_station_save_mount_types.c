/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/oskar_station.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_save_mount_types(const oskar_Station* station,
        const char* filename, int* status)
{
    FILE* file = 0;
    int i = 0;
    const char* type = 0;
    if (*status || !station) return;
    file = fopen(filename, "w");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    type = oskar_station_element_mount_types_const(station);
    const int num_elements = oskar_station_num_elements(station);
    for (i = 0; i < num_elements; ++i) fprintf(file, "%c\n", type[i]);
    fclose(file);
}

#ifdef __cplusplus
}
#endif
