/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_random_gaussian.h"
#include "sky/oskar_sky.h"
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Sky* oskar_sky_generate_grid(int precision, double ra0_rad,
        double dec0_rad, int side_length, double fov_rad, double mean_flux_jy,
        double std_flux_jy, int seed, int* status)
{
    oskar_Sky* sky = 0;
    double r[2];
    if (*status) return 0;

    /* Create a temporary sky model. */
    const int num_points = side_length * side_length;
    sky = oskar_sky_create(precision, OSKAR_CPU, num_points, status);

    /* Side length of 1 is a special case. */
    if (side_length == 1)
    {
        /* Generate the Stokes I flux and store the value. */
        oskar_random_gaussian2(seed, 0, 0, r);
        r[0] = mean_flux_jy + std_flux_jy * r[0];
        oskar_sky_set_source(sky, 0, ra0_rad, dec0_rad, r[0], 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
    }
    else
    {
        int i = 0, j = 0, k = 0;
        const double l_max = sin(0.5 * fov_rad);
        const double sin_dec0 = sin(dec0_rad);
        const double cos_dec0 = cos(dec0_rad);
        for (j = 0, k = 0; j < side_length; ++j)
        {
            const double m = 2.0 * l_max * j / (side_length - 1) - l_max;
            for (i = 0; i < side_length; ++i, ++k)
            {
                const double l = -2.0 * l_max * i / (side_length - 1) + l_max;

                /* Get longitude and latitude from tangent plane coords. */
                const double n = sqrt(1.0 - l*l - m*m);
                const double dec = asin(n * sin_dec0 + m * cos_dec0);
                const double ra = atan2(l, cos_dec0 * n - m * sin_dec0);

                /* Generate the Stokes I flux and store the value. */
                oskar_random_gaussian2(seed, i, j, r);
                r[0] = mean_flux_jy + std_flux_jy * r[0];
                oskar_sky_set_source(sky, k, ra + ra0_rad, dec, r[0],
                        0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
            }
        }
    }

    return sky;
}

#ifdef __cplusplus
}
#endif
