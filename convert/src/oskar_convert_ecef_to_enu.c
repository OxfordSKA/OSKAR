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

#include <oskar_convert_ecef_to_enu.h>
#include <oskar_convert_geodetic_spherical_to_ecef.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_convert_ecef_to_enu(int num_points, const double* ecef_x,
        const double* ecef_y, const double* ecef_z, double lon_rad,
        double lat_rad, double alt_metres, double* x, double* y, double* z)
{
    int i;
    double x0, y0, z0, a, b, c, d;
    double sin_lon, cos_lon, sin_lat, cos_lat;

    /* Get ECEF coordinates of reference position. */
    oskar_convert_geodetic_spherical_to_ecef(1,
            &lon_rad, &lat_rad, &alt_metres, &x0, &y0, &z0);

    /* Get rotation matrix elements. */
    sin_lon = sin(lon_rad);
    cos_lon = cos(lon_rad);
    sin_lat = sin(lat_rad);
    cos_lat = cos(lat_rad);
    a = -sin_lat * cos_lon;
    b = -sin_lat * sin_lon;
    c = cos_lat * cos_lon;
    d = cos_lat * sin_lon;

    /* Loop over points. */
    for (i = 0; i < num_points; ++i)
    {
        /* Get deltas from reference point. */
        double dx, dy, dz;
        dx = ecef_x[i] - x0;
        dy = ecef_y[i] - y0;
        dz = ecef_z[i] - z0;

        /* Get horizon coordinates. */
        x[i] = -sin_lon * dx + cos_lon * dy;
        y[i] = a * dx + b * dy + cos_lat * dz;
        z[i] = c * dx + d * dy + sin_lat * dz;
    }
}

#ifdef __cplusplus
}
#endif
