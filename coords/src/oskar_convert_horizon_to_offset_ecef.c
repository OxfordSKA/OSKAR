/*
 * Copyright (c) 2013, The University of Oxford
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

#include "oskar_convert_horizon_to_offset_ecef.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Double precision. */
void oskar_convert_horizon_to_offset_ecef_d(int n, const double* horizon_x,
        const double* horizon_y, const double* horizon_z, double lon, double lat,
        double* offset_ecef_x, double* offset_ecef_y, double* offset_ecef_z)
{
    /* Precompute some trig. */
    double sin_l, cos_l, sin_p, cos_p;
    int i;
    sin_l = sin(lon);
    cos_l = cos(lon);
    sin_p = sin(lat);
    cos_p = cos(lat);

    /* Loop over points. */
    for (i = 0; i < n; ++i)
    {
        double xi, yi, zi, xt, yt, zt;

        /* Get the input coordinates. */
        xi = horizon_x[i];
        yi = horizon_y[i];
        zi = horizon_z[i];

        /* Apply rotation matrix. */
        xt = -xi * sin_l - yi * sin_p * cos_l + zi * cos_p * cos_l;
        yt =  xi * cos_l - yi * sin_p * sin_l + zi * cos_p * sin_l;
        zt =  yi * cos_p + zi * sin_p;

        /* Save the rotated values. */
        offset_ecef_x[i] = xt;
        offset_ecef_y[i] = yt;
        offset_ecef_z[i] = zt;
    }
}

/* Single precision. */
void oskar_convert_horizon_to_offset_ecef_f(int n, const float* horizon_x,
        const float* horizon_y, const float* horizon_z, float lon, float lat,
        float* offset_ecef_x, float* offset_ecef_y, float* offsec_ecef_z)
{
    /* Precompute some trig. */
    double sin_l, cos_l, sin_p, cos_p;
    int i;
    sin_l = sin(lon);
    cos_l = cos(lon);
    sin_p = sin(lat);
    cos_p = cos(lat);

    /* Loop over points. */
    for (i = 0; i < n; ++i)
    {
        double xi, yi, zi, xt, yt, zt;

        /* Get the input coordinates. */
        xi = (double) (horizon_x[i]);
        yi = (double) (horizon_y[i]);
        zi = (double) (horizon_z[i]);

        /* Apply rotation matrix. */
        xt = -xi * sin_l - yi * sin_p * cos_l + zi * cos_p * cos_l;
        yt =  xi * cos_l - yi * sin_p * sin_l + zi * cos_p * sin_l;
        zt =  yi * cos_p + zi * sin_p;

        /* Save the rotated values. */
        offset_ecef_x[i] = (float)xt;
        offset_ecef_y[i] = (float)yt;
        offsec_ecef_z[i] = (float)zt;
    }
}

#ifdef __cplusplus
}
#endif
