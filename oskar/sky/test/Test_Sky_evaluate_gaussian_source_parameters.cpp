/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "math/oskar_cmath.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_timer.h"

#define DEG2RAD (M_PI / 180.0)
#define ARCSEC2RAD (DEG2RAD / 3600.0)


TEST(Sky, evaluate_gaussian_source_parameters)
{
    int num_sources = 16384;

    // Run in both single and double precision.
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    for (int i_type = 0; i_type < 2; ++i_type)
    {
        int status = 0;
        int num_failed = 0;
        oskar_Sky* sky = oskar_sky_create(
                types[i_type], OSKAR_CPU, num_sources, &status
        );
        for (int i = 0; i < num_sources; ++i)
        {
            oskar_sky_set_source(sky, i,
                    i * DEG2RAD * 0.001, i * DEG2RAD * 0.001,
                    0., 0., 0., 0., 0., 0., 0.,
                    1200. * ARCSEC2RAD, 600. * ARCSEC2RAD, 30. * DEG2RAD,
                    &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
        oskar_Timer* tmr = oskar_timer_create(OSKAR_TIMER_NATIVE);
        oskar_timer_start(tmr);
        oskar_sky_evaluate_gaussian_source_parameters(
                sky, false, 0., 10. * DEG2RAD, &num_failed, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(0, num_failed);
        printf("Evaluate Gaussian source parameters took %.3f s\n",
                oskar_timer_elapsed(tmr)
        );
        oskar_sky_free(sky, &status);
        oskar_timer_free(tmr);
    }
}
