/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"
#include "math/oskar_cmath.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define R2D (180.0 / M_PI)

void oskar_station_save_feed_angle(const oskar_Station* station,
        int feed, const char* filename, int* status)
{
    int i = 0;
    FILE* file = 0;
    const double *a = 0, *b = 0, *c = 0;
    if (*status || !station) return;
    a = oskar_mem_double_const(station->element_euler_cpu[feed][0], status);
    b = oskar_mem_double_const(station->element_euler_cpu[feed][1], status);
    c = oskar_mem_double_const(station->element_euler_cpu[feed][2], status);
    file = fopen(filename, "w");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    const int num_elements = oskar_station_num_elements(station);
    for (i = 0; i < num_elements; ++i)
    {
        fprintf(file, "% 14.6f % 14.6f % 14.6f\n",
                a[i] * R2D, b[i] * R2D, c[i] * R2D);
    }
    fclose(file);
}

#ifdef __cplusplus
}
#endif
