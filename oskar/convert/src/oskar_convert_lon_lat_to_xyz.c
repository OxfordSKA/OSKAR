/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_lon_lat_to_xyz.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_lon_lat_to_xyz_f(int num_points, const float* lon_rad,
        const float* lat_rad, float* x, float* y, float* z)
{
    int i = 0;
    for (i = 0; i < num_points; ++i)
    {
        const float cos_lon = cosf(lon_rad[i]);
        const float sin_lon = sinf(lon_rad[i]);
        const float cos_lat = cosf(lat_rad[i]);
        const float sin_lat = sinf(lat_rad[i]);
        x[i] = cos_lat * cos_lon;
        y[i] = cos_lat * sin_lon;
        z[i] = sin_lat;
    }
}

void oskar_convert_lon_lat_to_xyz_d(int num_points, const double* lon_rad,
        const double* lat_rad, double* x, double* y, double* z)
{
    int i = 0;
    for (i = 0; i < num_points; ++i)
    {
        const double cos_lon = cos(lon_rad[i]);
        const double sin_lon = sin(lon_rad[i]);
        const double cos_lat = cos(lat_rad[i]);
        const double sin_lat = sin(lat_rad[i]);
        x[i] = cos_lat * cos_lon;
        y[i] = cos_lat * sin_lon;
        z[i] = sin_lat;
    }
}

#ifdef __cplusplus
}
#endif
