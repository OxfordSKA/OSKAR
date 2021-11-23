/*
 * Copyright (c) 2014-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/oskar_station.h"
#include "telescope/station/private_station.h"

#include "utility/oskar_getline.h"
#include "utility/oskar_string_to_array.h"
#include "math/oskar_cmath.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_save_permitted_beams(const oskar_Station* station,
        const char* filename, int* status)
{
    FILE* file = 0;
    const double *az = 0, *el = 0;
    int i = 0;
    if (*status || !station) return;
    file = fopen(filename, "w");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    az = oskar_mem_double_const(
            oskar_station_permitted_beam_az_rad_const(station), status);
    el = oskar_mem_double_const(
            oskar_station_permitted_beam_el_rad_const(station), status);
    const int num_beams = oskar_station_num_permitted_beams(station);
    for (i = 0; i < num_beams; ++i)
    {
        fprintf(file, "%.6f %.6f\n", az[i] * 180.0 / M_PI,
                el[i] * 180.0 / M_PI);
    }
    fclose(file);
}

#ifdef __cplusplus
}
#endif
