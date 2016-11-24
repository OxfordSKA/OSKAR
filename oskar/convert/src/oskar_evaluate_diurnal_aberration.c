/*
 * Copyright (c) 2014, The University of Oxford
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

#include "convert/oskar_evaluate_diurnal_aberration.h"
#include "convert/oskar_convert_geodetic_spherical_to_ecef.h"
#include "math/oskar_cmath.h"

#ifdef __cplusplus
extern "C" {
#endif

double oskar_evaluate_diurnal_aberration(double lon_rad, double lat_rad,
        double height_m, double era_rad, double pm_x_rad, double pm_y_rad,
        double s_prime)
{
    double sin_sp, cos_sp, sin_xp, cos_xp, sin_yp, cos_yp, sin_era, cos_era;
    double a, b, x, y, z, vx, vy;

    /* Earth rotation rate in rad per second of UT1. */
    const double omega_earth = 1.00273781191135448 * 2.0 * M_PI / 86400.0;

    /* Transform geodetic (longitude, latitude, altitude) to geocentric XYZ. */
    oskar_convert_geodetic_spherical_to_ecef(1,
            &lon_rad, &lat_rad, &height_m, &x, &y, &z);

    /* Calculate elements of polar motion matrix. */
    sin_sp = sin(s_prime);
    cos_sp = cos(s_prime);
    sin_xp = sin(pm_x_rad);
    cos_xp = cos(pm_x_rad);
    sin_yp = sin(pm_y_rad);
    cos_yp = cos(pm_y_rad);

    /* Matrix-vector multiplication. */
    a = x * cos_sp * cos_xp +
            y * (cos_yp * sin_sp + cos_sp * sin_xp * sin_yp) +
            z * (cos_sp * cos_yp * sin_xp - sin_sp * sin_yp);
    b = x * -cos_xp * sin_sp +
            y * (cos_sp * cos_yp - sin_sp * sin_xp * sin_yp) +
            z * (-cos_sp * sin_yp - cos_yp * sin_sp * sin_xp);

    /* Get velocity components. */
    sin_era = sin(era_rad);
    cos_era = cos(era_rad);
    vx = omega_earth * (-sin_era * a - cos_era * b);
    vy = omega_earth * ( cos_era * a - sin_era * b);

    /* Return magnitude of diurnal aberration vector. */
    return sqrt(vx*vx + vy*vy) / 299792458.0;
}

#ifdef __cplusplus
}
#endif
