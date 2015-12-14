/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <oskar_sky.h>

#include <oskar_fit_ellipse.h>
#include <oskar_rotate.h>
#include <oskar_convert_lon_lat_to_relative_directions.h>
#include <oskar_convert_lon_lat_to_xyz.h>
#include <oskar_convert_relative_directions_to_lon_lat.h>
#include <oskar_convert_xyz_to_lon_lat.h>

#include <stdlib.h>
#include <oskar_cmath.h>

#define M_PI_2_2_LN_2 7.11941466249375271693034 /* pi^2 / (2 log_e(2)) */
#define MAX(x, y) (((x) > (y)) ? (x) : (y))
#define MIN(x, y) (((x) < (y)) ? (x) : (y))

#ifdef __cplusplus
extern "C" {
#endif

/* Number of points that define the ellipse */
#define ELLIPSE_PTS 6

void oskar_sky_evaluate_gaussian_source_parameters(oskar_Sky* sky,
        int zero_failed_sources, double ra0, double dec0, int* num_failed,
        int* status)
{
    int i, j, num_sources;
    int type;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Return if memory is not on the CPU. */
    if (oskar_sky_mem_location(sky) != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Get data type and number of sources. */
    type = oskar_sky_precision(sky);
    num_sources = oskar_sky_num_sources(sky);

    /* Switch on type. */
    if (type == OSKAR_DOUBLE)
    {
        /* Double precision. */
        const double *ra_, *dec_, *maj_, *min_, *pa_;
        double *I_, *Q_, *U_, *V_, *a_, *b_, *c_;
        double cos_pa_2, sin_pa_2, sin_2pa, inv_std_min_2, inv_std_maj_2;
        double ellipse_a, ellipse_b, maj, min, pa, cos_pa, sin_pa, t;
        double l[ELLIPSE_PTS], m[ELLIPSE_PTS];
        double work1[5 * ELLIPSE_PTS], work2[5 * ELLIPSE_PTS];
        double lon[ELLIPSE_PTS], lat[ELLIPSE_PTS];
        double x[ELLIPSE_PTS], y[ELLIPSE_PTS], z[ELLIPSE_PTS];
        ra_  = oskar_mem_double_const(oskar_sky_ra_rad_const(sky), status);
        dec_ = oskar_mem_double_const(oskar_sky_dec_rad_const(sky), status);
        maj_ = oskar_mem_double_const(oskar_sky_fwhm_major_rad_const(sky), status);
        min_ = oskar_mem_double_const(oskar_sky_fwhm_minor_rad_const(sky), status);
        pa_  = oskar_mem_double_const(oskar_sky_position_angle_rad_const(sky), status);
        I_   = oskar_mem_double(oskar_sky_I(sky), status);
        Q_   = oskar_mem_double(oskar_sky_Q(sky), status);
        U_   = oskar_mem_double(oskar_sky_U(sky), status);
        V_   = oskar_mem_double(oskar_sky_V(sky), status);
        a_   = oskar_mem_double(oskar_sky_gaussian_a(sky), status);
        b_   = oskar_mem_double(oskar_sky_gaussian_b(sky), status);
        c_   = oskar_mem_double(oskar_sky_gaussian_c(sky), status);

        for (i = 0; i < num_sources; ++i)
        {
            /* Note: could do something different from the projection below
             * in the case of a line (i.e. maj or min = 0), as in this case
             * there is no ellipse to project, only two points.
             * -- This continue could then be a if() .. else() instead.
             */
            if (maj_[i] == 0.0 && min_[i] == 0.0) continue;

            /* Evaluate shape of ellipse on the l,m plane. */
            ellipse_a = maj_[i]/2.0;
            ellipse_b = min_[i]/2.0;
            cos_pa = cos(pa_[i]);
            sin_pa = sin(pa_[i]);
            for (j = 0; j < ELLIPSE_PTS; ++j)
            {
                t = j * 60.0 * M_PI / 180.0;
                l[j] = ellipse_a*cos(t)*sin_pa + ellipse_b*sin(t)*cos_pa;
                m[j] = ellipse_a*cos(t)*cos_pa - ellipse_b*sin(t)*sin_pa;
            }
            oskar_convert_relative_directions_to_lon_lat_2d_d(ELLIPSE_PTS,
                    l, m, 0.0, 0.0, lon, lat);

            /* Rotate on the sphere. */
            oskar_convert_lon_lat_to_xyz_d(ELLIPSE_PTS, lon, lat, x, y, z);
            oskar_rotate_sph_d(ELLIPSE_PTS, x, y, z, ra_[i], dec_[i]);
            oskar_convert_xyz_to_lon_lat_d(ELLIPSE_PTS, x, y, z, lon, lat);

            oskar_convert_lon_lat_to_relative_directions_2d_d(
                    ELLIPSE_PTS, lon, lat, ra0, dec0, l, m);

            /* Get new major and minor axes and position angle. */
            oskar_fit_ellipse_d(&maj, &min, &pa, ELLIPSE_PTS, l, m, work1,
                    work2, status);

            /* Check if fitting failed. */
            if (*status == OSKAR_ERR_ELLIPSE_FIT_FAILED)
            {
                if (zero_failed_sources)
                {
                    I_[i] = 0.0;
                    Q_[i] = 0.0;
                    U_[i] = 0.0;
                    V_[i] = 0.0;
                }
                ++(*num_failed);
                *status = 0;
                continue;
            }
            else if (*status) break;

            /* Evaluate ellipse parameters. */
            inv_std_maj_2 = 0.5 * (maj * maj) * M_PI_2_2_LN_2;
            inv_std_min_2 = 0.5 * (min * min) * M_PI_2_2_LN_2;
            cos_pa_2 = cos(pa) * cos(pa);
            sin_pa_2 = sin(pa) * sin(pa);
            sin_2pa  = sin(2.0 * pa);
            a_[i] = cos_pa_2*inv_std_min_2     + sin_pa_2*inv_std_maj_2;
            b_[i] = -sin_2pa*inv_std_min_2*0.5 + sin_2pa *inv_std_maj_2*0.5;
            c_[i] = sin_pa_2*inv_std_min_2     + cos_pa_2*inv_std_maj_2;
        }
    }
    else
    {
        /* Single precision. */
        const float *ra_, *dec_, *maj_, *min_, *pa_;
        float *I_, *Q_, *U_, *V_, *a_, *b_, *c_;
        float cos_pa_2, sin_pa_2, sin_2pa, inv_std_min_2, inv_std_maj_2;
        float ellipse_a, ellipse_b, maj, min, pa, cos_pa, sin_pa, t;
        float l[ELLIPSE_PTS], m[ELLIPSE_PTS];
        float work1[5 * ELLIPSE_PTS], work2[5 * ELLIPSE_PTS];
        float lon[ELLIPSE_PTS], lat[ELLIPSE_PTS];
        float x[ELLIPSE_PTS], y[ELLIPSE_PTS], z[ELLIPSE_PTS];
        ra_  = oskar_mem_float_const(oskar_sky_ra_rad_const(sky), status);
        dec_ = oskar_mem_float_const(oskar_sky_dec_rad_const(sky), status);
        maj_ = oskar_mem_float_const(oskar_sky_fwhm_major_rad_const(sky), status);
        min_ = oskar_mem_float_const(oskar_sky_fwhm_minor_rad_const(sky), status);
        pa_  = oskar_mem_float_const(oskar_sky_position_angle_rad_const(sky), status);
        I_   = oskar_mem_float(oskar_sky_I(sky), status);
        Q_   = oskar_mem_float(oskar_sky_Q(sky), status);
        U_   = oskar_mem_float(oskar_sky_U(sky), status);
        V_   = oskar_mem_float(oskar_sky_V(sky), status);
        a_   = oskar_mem_float(oskar_sky_gaussian_a(sky), status);
        b_   = oskar_mem_float(oskar_sky_gaussian_b(sky), status);
        c_   = oskar_mem_float(oskar_sky_gaussian_c(sky), status);

        for (i = 0; i < num_sources; ++i)
        {
            /* Note: could do something different from the projection below
             * in the case of a line (i.e. maj or min = 0), as in this case
             * there is no ellipse to project, only two points.
             * -- This continue could then be a if() .. else() instead.
             */
            if (maj_[i] == 0.0 && min_[i] == 0.0) continue;

            /* Evaluate shape of ellipse on the l,m plane. */
            ellipse_a = maj_[i]/2.0;
            ellipse_b = min_[i]/2.0;
            cos_pa = cos(pa_[i]);
            sin_pa = sin(pa_[i]);
            for (j = 0; j < ELLIPSE_PTS; ++j)
            {
                t = j * 60.0 * M_PI / 180.0;
                l[j] = ellipse_a*cos(t)*sin_pa + ellipse_b*sin(t)*cos_pa;
                m[j] = ellipse_a*cos(t)*cos_pa - ellipse_b*sin(t)*sin_pa;
            }
            oskar_convert_relative_directions_to_lon_lat_2d_f(ELLIPSE_PTS,
                    l, m, 0.0, 0.0, lon, lat);

            /* Rotate on the sphere. */
            oskar_convert_lon_lat_to_xyz_f(ELLIPSE_PTS, lon, lat, x, y, z);
            oskar_rotate_sph_f(ELLIPSE_PTS, x, y, z, ra_[i], dec_[i]);
            oskar_convert_xyz_to_lon_lat_f(ELLIPSE_PTS, x, y, z, lon, lat);

            oskar_convert_lon_lat_to_relative_directions_2d_f(
                    ELLIPSE_PTS, lon, lat, (float)ra0, (float)dec0, l, m);

            /* Get new major and minor axes and position angle. */
            oskar_fit_ellipse_f(&maj, &min, &pa, ELLIPSE_PTS, l, m, work1,
                    work2, status);

            /* Check if fitting failed. */
            if (*status == OSKAR_ERR_ELLIPSE_FIT_FAILED)
            {
                if (zero_failed_sources)
                {
                    I_[i] = 0.0;
                    Q_[i] = 0.0;
                    U_[i] = 0.0;
                    V_[i] = 0.0;
                }
                ++(*num_failed);
                *status = 0;
                continue;
            }
            else if (*status) break;

            /* Evaluate ellipse parameters. */
            inv_std_maj_2 = 0.5 * (maj * maj) * M_PI_2_2_LN_2;
            inv_std_min_2 = 0.5 * (min * min) * M_PI_2_2_LN_2;
            cos_pa_2 = cos(pa) * cos(pa);
            sin_pa_2 = sin(pa) * sin(pa);
            sin_2pa  = sin(2.0 * pa);
            a_[i] = cos_pa_2*inv_std_min_2     + sin_pa_2*inv_std_maj_2;
            b_[i] = -sin_2pa*inv_std_min_2*0.5 + sin_2pa *inv_std_maj_2*0.5;
            c_[i] = sin_pa_2*inv_std_min_2     + cos_pa_2*inv_std_maj_2;
        }
    }
}

#ifdef __cplusplus
}
#endif
