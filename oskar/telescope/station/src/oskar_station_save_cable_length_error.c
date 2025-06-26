/*
 * Copyright (c) 2019-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_save_cable_length_error(const oskar_Station* station,
        int feed, const char* filename, int* status)
{
    FILE* file = 0;
    if (*status || !station) return;
    file = fopen(filename, "w");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    oskar_mem_save_ascii(file, 1, 0, oskar_station_num_elements(station),
            status,
            oskar_station_element_cable_length_error_metres_const(station, feed));
    (void) fclose(file);
}

#ifdef __cplusplus
}
#endif
