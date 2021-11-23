/*
 * Copyright (c) 2016-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"
#include "sky/oskar_generate_random_coordinate.h"
#include "math/oskar_random_power_law.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Sky* oskar_sky_generate_random_power_law(int precision, int num_sources,
        double flux_min_jy, double flux_max_jy, double power, int seed,
        int* status)
{
    oskar_Sky* sky = 0;
    int i = 0;
    if (*status) return 0;

    /* Create a temporary sky model. */
    srand(seed);
    sky = oskar_sky_create(precision, OSKAR_CPU, num_sources, status);
    for (i = 0; i < num_sources; ++i)
    {
        double ra = 0.0, dec = 0.0, b = 0.0;
        oskar_generate_random_coordinate(&ra, &dec);
        b = oskar_random_power_law(flux_min_jy, flux_max_jy, power);
        oskar_sky_set_source(sky, i, ra, dec, b, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, status);
    }

    return sky;
}

#ifdef __cplusplus
}
#endif
