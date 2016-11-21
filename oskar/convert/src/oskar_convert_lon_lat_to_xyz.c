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

#include <oskar_convert_lon_lat_to_xyz.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_lon_lat_to_xyz_f(int num_points, const float* lon_rad,
        const float* lat_rad, float* x, float* y, float* z)
{
    int i;
    float cos_lon, sin_lon, cos_lat, sin_lat;
    for (i = 0; i < num_points; ++i)
    {
        cos_lon = cosf(lon_rad[i]);
        sin_lon = sinf(lon_rad[i]);
        cos_lat = cosf(lat_rad[i]);
        sin_lat = sinf(lat_rad[i]);

        x[i] = cos_lat * cos_lon;
        y[i] = cos_lat * sin_lon;
        z[i] = sin_lat;
    }
}

void oskar_convert_lon_lat_to_xyz_d(int num_points, const double* lon_rad,
        const double* lat_rad, double* x, double* y, double* z)
{
    int i;
    double cos_lon, sin_lon, cos_lat, sin_lat;
    for (i = 0; i < num_points; ++i)
    {
        cos_lon = cos(lon_rad[i]);
        sin_lon = sin(lon_rad[i]);
        cos_lat = cos(lat_rad[i]);
        sin_lat = sin(lat_rad[i]);

        x[i] = cos_lat * cos_lon;
        y[i] = cos_lat * sin_lon;
        z[i] = sin_lat;
    }
}


#ifdef __cplusplus
}
#endif
