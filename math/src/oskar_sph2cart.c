/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sph2cart_f(int num_points, float* x, float* y, float* z,
        const float* lon, const float* lat)
{
    int i;
    float cosLon, sinLon, cosLat, sinLat;
    for (i = 0; i < num_points; ++i)
    {
        cosLon = cosf(lon[i]);
        sinLon = sinf(lon[i]);
        cosLat = cosf(lat[i]);
        sinLat = sinf(lat[i]);

        x[i] = cosLat * cosLon;
        y[i] = cosLat * sinLon;
        z[i] = sinLat;
    }
}

void oskar_sph2cart_d(int num_points, double* x, double* y, double* z,
        const double* lon, const double* lat)
{
    int i;
    double cosLon, sinLon, cosLat, sinLat;
    for (i = 0; i < num_points; ++i)
    {
        cosLon = cos(lon[i]);
        sinLon = sin(lon[i]);
        cosLat = cos(lat[i]);
        sinLat = sin(lat[i]);

        x[i] = cosLat * cosLon;
        y[i] = cosLat * sinLon;
        z[i] = sinLat;
    }
}

#ifdef __cplusplus
}
#endif
