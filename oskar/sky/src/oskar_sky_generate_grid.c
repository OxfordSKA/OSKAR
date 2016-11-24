/*
 * Copyright (c) 2016, The University of Oxford
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

#include "sky/oskar_sky.h"
#include "math/oskar_random_gaussian.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Sky* oskar_sky_generate_grid(int precision, double ra0_rad,
        double dec0_rad, int side_length, double fov_rad, double mean_flux_jy,
        double std_flux_jy, int seed, int* status)
{
    oskar_Sky* t = 0;
    int i, j, k, num_points;
    double r[2];

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Create a temporary sky model. */
    num_points = side_length * side_length;
    t = oskar_sky_create(precision, OSKAR_CPU, num_points, status);

    /* Side length of 1 is a special case. */
    if (side_length == 1)
    {
        /* Generate the Stokes I flux and store the value. */
        oskar_random_gaussian2(seed, 0, 0, r);
        r[0] = mean_flux_jy + std_flux_jy * r[0];
        oskar_sky_set_source(t, 0, ra0_rad, dec0_rad, r[0], 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
    }
    else
    {
        double l_max, l, m, n, sin_dec0, cos_dec0, ra, dec;
        l_max = sin(0.5 * fov_rad);
        sin_dec0 = sin(dec0_rad);
        cos_dec0 = cos(dec0_rad);
        for (j = 0, k = 0; j < side_length; ++j)
        {
            m = 2.0 * l_max * j / (side_length - 1) - l_max;
            for (i = 0; i < side_length; ++i, ++k)
            {
                l = -2.0 * l_max * i / (side_length - 1) + l_max;

                /* Get longitude and latitude from tangent plane coords. */
                n = sqrt(1.0 - l*l - m*m);
                dec = asin(n * sin_dec0 + m * cos_dec0);
                ra = ra0_rad + atan2(l, cos_dec0 * n - m * sin_dec0);

                /* Generate the Stokes I flux and store the value. */
                oskar_random_gaussian2(seed, i, j, r);
                r[0] = mean_flux_jy + std_flux_jy * r[0];
                oskar_sky_set_source(t, k, ra, dec, r[0], 0.0, 0.0,
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
            }
        }
    }

    return t;
}

#ifdef __cplusplus
}
#endif
