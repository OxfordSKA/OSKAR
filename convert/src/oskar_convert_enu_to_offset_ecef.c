/*
 * Copyright (c) 2013-2014, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#include <oskar_convert_enu_to_offset_ecef.h>
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
    double sin_lon, cos_lon, sin_lat, cos_lat;
    int i;
    sin_lon = sin(lon_rad);
    cos_lon = cos(lon_rad);
    sin_lat = sin(lat_rad);
    cos_lat = cos(lat_rad);

    /* Loop over points. */
    for (i = 0; i < num_points; ++i)
    {
        double xi, yi, zi, xt, yt, zt;

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
    double sin_lon, cos_lon, sin_lat, cos_lat;
    int i;
    sin_lon = sin(lon_rad);
    cos_lon = cos(lon_rad);
    sin_lat = sin(lat_rad);
    cos_lat = cos(lat_rad);

    /* Loop over points. */
    for (i = 0; i < num_points; ++i)
    {
        double xi, yi, zi, xt, yt, zt;

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
