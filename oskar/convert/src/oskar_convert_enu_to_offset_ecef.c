/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "convert/oskar_convert_enu_to_offset_ecef.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Double precision. */
void oskar_convert_enu_to_offset_ecef_d(int num_points,
        const double* horizon_x, const double* horizon_y,
        const double* horizon_z, double lon_rad, double lat_rad,
        double* offset_ecef_x, double* offset_ecef_y, double* offset_ecef_z)
{
    /* Precompute some trig. */
    int i = 0;
    const double sin_lon = sin(lon_rad);
    const double cos_lon = cos(lon_rad);
    const double sin_lat = sin(lat_rad);
    const double cos_lat = cos(lat_rad);

    /* Loop over points. */
    for (i = 0; i < num_points; ++i)
    {
        double xi = 0.0, yi = 0.0, zi = 0.0, xt = 0.0, yt = 0.0, zt = 0.0;

        /* Get the input coordinates. */
        xi = horizon_x[i];
        yi = horizon_y[i];
        zi = horizon_z[i];

        /* Apply rotation matrix. */
        xt = -xi * sin_lon - yi * sin_lat * cos_lon + zi * cos_lat * cos_lon;
        yt =  xi * cos_lon - yi * sin_lat * sin_lon + zi * cos_lat * sin_lon;
        zt =  yi * cos_lat + zi * sin_lat;

        /* Save the rotated values. */
        offset_ecef_x[i] = xt;
        offset_ecef_y[i] = yt;
        offset_ecef_z[i] = zt;
    }
}

/* Single precision. */
void oskar_convert_enu_to_offset_ecef_f(int num_points,
        const float* horizon_x, const float* horizon_y,
        const float* horizon_z, float lon_rad, float lat_rad,
        float* offset_ecef_x, float* offset_ecef_y, float* offsec_ecef_z)
{
    /* Precompute some trig. */
    int i = 0;
    const double sin_lon = sin((double)lon_rad);
    const double cos_lon = cos((double)lon_rad);
    const double sin_lat = sin((double)lat_rad);
    const double cos_lat = cos((double)lat_rad);

    /* Loop over points. */
    for (i = 0; i < num_points; ++i)
    {
        double xi = 0.0, yi = 0.0, zi = 0.0, xt = 0.0, yt = 0.0, zt = 0.0;

        /* Get the input coordinates. */
        xi = (double) (horizon_x[i]);
        yi = (double) (horizon_y[i]);
        zi = (double) (horizon_z[i]);

        /* Apply rotation matrix. */
        xt = -xi * sin_lon - yi * sin_lat * cos_lon + zi * cos_lat * cos_lon;
        yt =  xi * cos_lon - yi * sin_lat * sin_lon + zi * cos_lat * sin_lon;
        zt =  yi * cos_lat + zi * sin_lat;

        /* Save the rotated values. */
        offset_ecef_x[i] = (float)xt;
        offset_ecef_y[i] = (float)yt;
        offsec_ecef_z[i] = (float)zt;
    }
}

#ifdef __cplusplus
}
#endif
