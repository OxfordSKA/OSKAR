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

#include "convert/oskar_convert_ecef_to_geodetic_spherical.h"
#include "convert/private_convert_ecef_to_geodetic_spherical_inline.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_ecef_to_geodetic_spherical_f(int num_points,
        const float* x, const float* y, const float* z,
        float* lon_rad, float* lat_rad, float* alt_m)
{
    int i;
    for (i = 0; i < num_points; ++i)
    {
        oskar_convert_ecef_to_geodetic_spherical_inline_f(x[i], y[i], z[i],
                &lon_rad[i], &lat_rad[i], &alt_m[i]);
    }
}

void oskar_convert_ecef_to_geodetic_spherical(int num_points,
        const double* x, const double* y, const double* z,
        double* lon_rad, double* lat_rad, double* alt_m)
{
    int i;
    for (i = 0; i < num_points; ++i)
    {
        oskar_convert_ecef_to_geodetic_spherical_inline_d(x[i], y[i], z[i],
                &lon_rad[i], &lat_rad[i], &alt_m[i]);
    }
}

#ifdef __cplusplus
}
#endif
