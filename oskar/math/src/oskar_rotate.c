/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_rotate.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_rotate_sph_f(int num_points, float* x, float* y, float* z,
        float lon, float lat)
{
    int i = 0;
    const float cos_lon = cosf(lon);
    const float sin_lon = sinf(lon);
    const float cos_lat = cosf(lat);
    const float sin_lat = sinf(lat);
    for (i = 0; i < num_points; ++i)
    {
        const float x_ = x[i];
        const float y_ = y[i];
        const float z_ = z[i];
        x[i] = x_ * cos_lon * cos_lat - y_ * sin_lon - z_ * cos_lon * sin_lat;
        y[i] = x_ * cos_lat * sin_lon + y_ * cos_lon - z_ * sin_lon * sin_lat;
        z[i] = x_ * sin_lat + z_ * cos_lat;
    }
}

void oskar_rotate_sph_d(int num_points, double* x, double* y, double* z,
        double lon, double lat)
{
    int i = 0;
    const double cos_lon = cos(lon);
    const double sin_lon = sin(lon);
    const double cos_lat = cos(lat);
    const double sin_lat = sin(lat);
    for (i = 0; i < num_points; ++i)
    {
        const double x_ = x[i];
        const double y_ = y[i];
        const double z_ = z[i];
        x[i] = x_ * cos_lon * cos_lat - y_ * sin_lon - z_ * cos_lon * sin_lat;
        y[i] = x_ * cos_lat * sin_lon + y_ * cos_lon - z_ * sin_lon * sin_lat;
        z[i] = x_ * sin_lat + z_ * cos_lat;
    }
}

#ifdef __cplusplus
}
#endif
