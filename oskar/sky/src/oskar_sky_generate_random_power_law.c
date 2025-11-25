/*
 * Copyright (c) 2016-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

 /* For srand() */
#include <stdlib.h>

#include "sky/oskar_sky.h"
#include "sky/oskar_generate_random_coordinate.h"
#include "math/oskar_random_power_law.h"

#ifdef __cplusplus
extern "C" {
#endif


oskar_Sky* oskar_sky_generate_random_power_law(
        int precision,
        int num_sources,
        double flux_min_jy,
        double flux_max_jy,
        double power,
        int seed,
        int* status
)
{
    oskar_Sky* sky = 0;
    oskar_Mem* col_ra = 0;
    oskar_Mem* col_dec = 0;
    oskar_Mem* col_flux = 0;
    int i = 0;
    if (*status) return 0;

    /* Create a temporary sky model and get handles to the relevant columns. */
    srand(seed);
    sky = oskar_sky_create(precision, OSKAR_CPU, num_sources, status);
    col_ra = oskar_sky_column(sky, OSKAR_SKY_RA_RAD, 0, status);
    col_dec = oskar_sky_column(sky, OSKAR_SKY_DEC_RAD, 0, status);
    col_flux = oskar_sky_column(sky, OSKAR_SKY_I_JY, 0, status);
    for (i = 0; i < num_sources; ++i)
    {
        double ra = 0.0, dec = 0.0, b = 0.0;
        oskar_generate_random_coordinate(&ra, &dec);
        b = oskar_random_power_law(flux_min_jy, flux_max_jy, power);
        oskar_mem_set_element_real(col_ra, i, ra, status);
        oskar_mem_set_element_real(col_dec, i, dec, status);
        oskar_mem_set_element_real(col_flux, i, b, status);
    }

    return sky;
}

#ifdef __cplusplus
}
#endif
