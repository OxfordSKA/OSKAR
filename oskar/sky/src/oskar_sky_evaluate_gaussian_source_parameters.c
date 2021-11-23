/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"

#include "math/oskar_fit_ellipse.h"
#include "math/oskar_rotate.h"
#include "convert/oskar_convert_lon_lat_to_relative_directions.h"
#include "convert/oskar_convert_lon_lat_to_xyz.h"
#include "convert/oskar_convert_relative_directions_to_lon_lat.h"
#include "convert/oskar_convert_xyz_to_lon_lat.h"

#include <stdlib.h>
#include "math/oskar_cmath.h"

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
    int i = 0, j = 0;
    if (*status) return;
    if (oskar_sky_mem_location(sky) != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }
    const int type = oskar_sky_precision(sky);
    const int num_sources = oskar_sky_num_sources(sky);
    const double sin_dec0 = sin(dec0), cos_dec0 = cos(dec0);
    if (type == OSKAR_DOUBLE)
    {
        /* Double precision. */
        const double *ra_ = 0, *dec_ = 0, *maj_ = 0, *min_ = 0, *pa_ = 0;
        double *I_ = 0, *Q_ = 0, *U_ = 0, *V_ = 0, *a_ = 0, *b_ = 0, *c_ = 0;
        double cos_pa_2 = 0.0, sin_pa_2 = 0.0, sin_2pa = 0.0;
        double inv_std_min_2 = 0.0, inv_std_maj_2 = 0.0;
        double ellipse_a = 0.0, ellipse_b = 0.0, maj = 0.0, min = 0.0, pa = 0.0;
        double cos_pa = 0.0, sin_pa = 0.0, t = 0.0;
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
#if 0
            /* TODO(FD) Consider replacing existing code with this version? */
            /* Get source parameters. */
            const double cos_ra = cos(ra_[i]);
            const double sin_ra = sin(ra_[i]);
            const double cos_dec = cos(dec_[i]);
            const double sin_dec = sin(dec_[i]);
            for (j = 0; j < ELLIPSE_PTS; ++j)
            {
                double sin_t, cos_t;
                /* Evaluate shape of ellipse on the l,m plane
                 * at RA = 0, Dec = 0. */
                t = j * 60.0 * M_PI / 180.0; sin_t = sin(t); cos_t = cos(t);
                const double l_ = ellipse_a * cos_t * sin_pa + ellipse_b * sin_t * cos_pa;
                const double m_ = ellipse_a * cos_t * cos_pa - ellipse_b * sin_t * sin_pa;
                const double n_ = sqrt(1.0 - l_*l_ - m_*m_);

                /* Rotate to source position.
                 * For a source on tangent plane at (x, y, z) = (1, 0, 0),
                 * x parallel to n, y parallel to l, z parallel to m. */
                const double x = n_ * cos_ra * cos_dec - l_ * sin_ra - m_ * cos_ra * sin_dec;
                const double y = n_ * cos_dec * sin_ra + l_ * cos_ra - m_ * sin_ra * sin_dec;
                const double z = n_ * sin_dec + m_ * cos_dec;

                /* Get ellipse points relative to phase centre.
                 * sin(lat) is z; cos(lat) is sqrt(x*x + y*y); t is rel_lon */
                t = atan2(y, x) - ra0; sin_t = sin(t); cos_t = cos(t);
                const double cos_lat = sqrt(x*x + y*y);
                l[j] = cos_lat * sin_t;
                m[j] = cos_dec0 * z - sin_dec0 * cos_lat * cos_t;
            }
#else
            for (j = 0; j < ELLIPSE_PTS; ++j)
            {
                t = j * 60.0 * M_PI / 180.0;
                l[j] = ellipse_a*cos(t)*sin_pa + ellipse_b*sin(t)*cos_pa;
                m[j] = ellipse_a*cos(t)*cos_pa - ellipse_b*sin(t)*sin_pa;
            }
            oskar_convert_relative_directions_to_lon_lat_2d_d(ELLIPSE_PTS,
                    l, m, 0, 0.0, 1.0, 0.0, lon, lat);

            /* Rotate on the sphere. */
            oskar_convert_lon_lat_to_xyz_d(ELLIPSE_PTS, lon, lat, x, y, z);
            oskar_rotate_sph_d(ELLIPSE_PTS, x, y, z, ra_[i], dec_[i]);
            oskar_convert_xyz_to_lon_lat_d(ELLIPSE_PTS, x, y, z, lon, lat);

            oskar_convert_lon_lat_to_relative_directions_2d_d(
                    ELLIPSE_PTS, lon, lat, ra0, cos_dec0, sin_dec0, l, m, 0);
#endif
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
        const float *ra_ = 0, *dec_ = 0, *maj_ = 0, *min_ = 0, *pa_ = 0;
        float *I_ = 0, *Q_ = 0, *U_ = 0, *V_ = 0, *a_ = 0, *b_ = 0, *c_ = 0;
        float cos_pa_2 = 0.0, sin_pa_2 = 0.0, sin_2pa = 0.0;
        float inv_std_min_2 = 0.0, inv_std_maj_2 = 0.0;
        float ellipse_a = 0.0, ellipse_b = 0.0, maj = 0.0, min = 0.0, pa = 0.0;
        float cos_pa = 0.0, sin_pa = 0.0, t = 0.0;
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
                    l, m, 0, 0.0, 1.0, 0.0, lon, lat);

            /* Rotate on the sphere. */
            oskar_convert_lon_lat_to_xyz_f(ELLIPSE_PTS, lon, lat, x, y, z);
            oskar_rotate_sph_f(ELLIPSE_PTS, x, y, z, ra_[i], dec_[i]);
            oskar_convert_xyz_to_lon_lat_f(ELLIPSE_PTS, x, y, z, lon, lat);

            oskar_convert_lon_lat_to_relative_directions_2d_f(ELLIPSE_PTS,
                    lon, lat, (float)ra0, (float)cos_dec0, (float)sin_dec0,
                    l, m, 0);

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
