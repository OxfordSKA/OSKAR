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

#include <oskar_convert_cirs_ra_dec_to_enu_directions.h>
#include <private_convert_cirs_relative_directions_to_enu_directions_inline.h>
#include <private_evaluate_cirs_observed_parameters.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_convert_cirs_ra_dec_to_enu_directions_f(int num_points,
        const float* ra_rad, const float* dec_rad, float lon_rad,
        float lat_rad, float era_rad, float pm_x_rad, float pm_y_rad,
        float diurnal_aberration, float* x, float* y, float* z)
{
    int i;
    double sin_lat, cos_lat, sin_ha0, cos_ha0, sin_dec0, cos_dec0;
    double local_pm_x, local_pm_y;

    /* Calculate common transform parameters. */
    oskar_evaluate_cirs_observed_parameters(lon_rad, lat_rad, era_rad,
            0.0, 0.0, pm_x_rad, pm_y_rad, &sin_lat, &cos_lat, &sin_ha0,
            &cos_ha0, &sin_dec0, &cos_dec0, &local_pm_x, &local_pm_y);

    /* Loop over positions. */
    for (i = 0; i < num_points; ++i)
    {
        float ra, dec, cos_dec, l, m, n;

        /* Get CIRS RA, Dec. */
        ra = ra_rad[i];
        dec = dec_rad[i];

        /* Convert to direction cosines (relative to RA=0, Dec=0). */
        cos_dec = cosf(dec);
        l = cos_dec * sinf(ra);
        m = sinf(dec);
        n = cos_dec * cosf(ra);

        /* Convert CIRS relative directions to ENU directions. */
        oskar_convert_cirs_relative_directions_to_enu_directions_inline_f(
                l, m, n, (float)sin_lat, (float)cos_lat, (float)sin_ha0,
                (float)cos_ha0, (float)sin_dec0, (float)cos_dec0,
                (float)local_pm_x, (float)local_pm_y, diurnal_aberration,
                &x[i], &y[i], &z[i]);
    }
}

/* Double precision. */
void oskar_convert_cirs_ra_dec_to_enu_directions_d(int num_points,
        const double* ra_rad, const double* dec_rad, double lon_rad,
        double lat_rad, double era_rad, double pm_x_rad, double pm_y_rad,
        double diurnal_aberration, double* x, double* y, double* z)
{
    int i;
    double sin_lat, cos_lat, sin_ha0, cos_ha0, sin_dec0, cos_dec0;
    double local_pm_x, local_pm_y;

    /* Calculate common transform parameters. */
    oskar_evaluate_cirs_observed_parameters(lon_rad, lat_rad, era_rad,
            0.0, 0.0, pm_x_rad, pm_y_rad, &sin_lat, &cos_lat, &sin_ha0,
            &cos_ha0, &sin_dec0, &cos_dec0, &local_pm_x, &local_pm_y);

    /* Loop over positions. */
    for (i = 0; i < num_points; ++i)
    {
        double ra, dec, cos_dec, l, m, n;

        /* Get CIRS RA, Dec. */
        ra = ra_rad[i];
        dec = dec_rad[i];

        /* Convert to direction cosines (relative to RA=0, Dec=0). */
        cos_dec = cos(dec);
        l = cos_dec * sin(ra);
        m = sin(dec);
        n = cos_dec * cos(ra);

        /* Convert CIRS relative directions to ENU directions. */
        oskar_convert_cirs_relative_directions_to_enu_directions_inline_d(
                l, m, n, sin_lat, cos_lat, sin_ha0, cos_ha0,
                sin_dec0, cos_dec0, local_pm_x, local_pm_y,
                diurnal_aberration, &x[i], &y[i], &z[i]);
    }
}

#ifdef __cplusplus
}
#endif
