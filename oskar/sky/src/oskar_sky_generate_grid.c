/*
 * Copyright (c) 2016-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <math.h>

#include "math/oskar_random_gaussian.h"
#include "sky/oskar_sky.h"

#ifdef __cplusplus
extern "C" {
#endif


oskar_Sky* oskar_sky_generate_grid(
        int precision,
        double ra0_rad,
        double dec0_rad,
        int side_length,
        double fov_rad,
        double mean_flux_jy,
        double std_flux_jy,
        int seed,
        int* status
)
{
    int i = 0, j = 0, k = 0;
    oskar_Sky* sky = 0;
    oskar_Mem* col_ra = 0;
    oskar_Mem* col_dec = 0;
    oskar_Mem* col_flux = 0;
    double r[2];
    const double l_max = sin(0.5 * fov_rad);
    const double sin_dec0 = sin(dec0_rad);
    const double cos_dec0 = cos(dec0_rad);
    const int num_points = side_length * side_length;
    if (*status) return 0;

    /* Create a temporary sky model and get handles to the relevant columns. */
    sky = oskar_sky_create(precision, OSKAR_CPU, num_points, status);
    col_ra = oskar_sky_column(sky, OSKAR_SKY_RA_RAD, 0, status);
    col_dec = oskar_sky_column(sky, OSKAR_SKY_DEC_RAD, 0, status);
    col_flux = oskar_sky_column(sky, OSKAR_SKY_I_JY, 0, status);

    /* Side length of 1 is a special case. */
    if (side_length == 1)
    {
        /* Generate the Stokes I flux and store the value. */
        oskar_random_gaussian2(seed, 0, 0, r);
        r[0] = mean_flux_jy + std_flux_jy * r[0];
        oskar_mem_set_element_real(col_ra, 0, ra0_rad, status);
        oskar_mem_set_element_real(col_dec, 0, dec0_rad, status);
        oskar_mem_set_element_real(col_flux, 0, r[0], status);
        return sky;
    }

    /* Generate grid. */
    for (j = 0, k = 0; j < side_length; ++j)
    {
        const double m = 2.0 * l_max * j / (side_length - 1) - l_max;
        for (i = 0; i < side_length; ++i, ++k)
        {
            const double l = -2.0 * l_max * i / (side_length - 1) + l_max;

            /* Get longitude and latitude from tangent plane coords. */
            const double n = sqrt(1.0 - l * l - m * m);
            const double dec = asin(n * sin_dec0 + m * cos_dec0);
            const double ra = ra0_rad + atan2(l, cos_dec0 * n - m * sin_dec0);

            /* Generate the Stokes I flux and store the value. */
            oskar_random_gaussian2(seed, i, j, r);
            r[0] = mean_flux_jy + std_flux_jy * r[0];
            oskar_mem_set_element_real(col_ra, k, ra, status);
            oskar_mem_set_element_real(col_dec, k, dec, status);
            oskar_mem_set_element_real(col_flux, k, r[0], status);
        }
    }
    return sky;
}

#ifdef __cplusplus
}
#endif
