/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_xyz_to_lon_lat.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_xyz_to_lon_lat_f(int num_points, const float* x,
        const float* y, const float* z, float* lon_rad, float* lat_rad)
{
    int i = 0;
    for (i = 0; i < num_points; ++i)
    {
        const float x_ = x[i];
        const float y_ = y[i];
        const float z_ = z[i];
        lon_rad[i] = atan2f(y_, x_);
        lat_rad[i] = atan2f(z_, sqrtf(x_*x_ + y_*y_));
    }
}

void oskar_convert_xyz_to_lon_lat_d(int num_points, const double* x,
        const double* y, const double* z, double* lon_rad, double* lat_rad)
{
    int i = 0;
    for (i = 0; i < num_points; ++i)
    {
        const double x_ = x[i];
        const double y_ = y[i];
        const double z_ = z[i];
        lon_rad[i] = atan2(y_, x_);
        lat_rad[i] = atan2(z_, sqrt(x_*x_ + y_*y_));
    }
}

#ifdef __cplusplus
}
#endif
