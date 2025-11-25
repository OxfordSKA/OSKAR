/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "math/oskar_cmath.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"


TEST(Sky, filter_by_radius)
{
    const int num_sources = 91;

    // Filter parameters, centred on the pole.
    const double ra0_rad = 0.0;
    const double dec0_rad = M_PI / 2;
    const double inner_radius_rad = 4.5 * (M_PI / 180);
    const double outer_radius_rad = 10.5 * (M_PI / 180);

    // Test both single and double precision.
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    for (int i_type = 0; i_type < 2; ++i_type)
    {
        // Generate a line of sources from dec = 0 to dec = 90 degrees.
        int status = 0;
        oskar_Sky* sky = oskar_sky_create(
                types[i_type], OSKAR_CPU, num_sources, &status
        );
        for (int i = 0; i < num_sources; ++i)
        {
            double dec = i * ((M_PI / 2) / (num_sources - 1));
            bool in_range = ((dec > dec0_rad - outer_radius_rad) &&
                    (dec <= dec0_rad - inner_radius_rad)
            );
            double value = in_range ? 1000.0 : 0.0;
            oskar_sky_set_source(
                    sky, i, 0.0, dec, value, value, value, value,
                    value, value, value, value, value, value, &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }

        // Apply radius filter.
        oskar_sky_filter_by_radius(
                sky, inner_radius_rad, outer_radius_rad,
                ra0_rad, dec0_rad, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the resulting sky model.
        ASSERT_EQ(6, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        for (int i = 0; i < oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES); ++i)
        {
            double dec = oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, i);
            ASSERT_GT(dec, 79.5 * M_PI / 180.0);
            ASSERT_LT(dec, 85.5 * M_PI / 180.0);
            for (int c = (int) OSKAR_SKY_Q_JY; c <= (int) OSKAR_SKY_PA_RAD; ++c)
            {
                EXPECT_EQ(
                    1000.0, oskar_sky_data(sky, (oskar_SkyColumn) c, 0, i)
                );
            }
        }

        // Free sky model.
        oskar_sky_free(sky, &status);
    }
}
