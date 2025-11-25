/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "math/oskar_cmath.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"


TEST(Sky, filter_by_flux)
{
    const int num_sources = 10000;
    const double flux_min = 4.99;
    const double flux_max = 10.01;
    const double inc = 0.05;
    int expected = 1 + (int) (round(flux_max / inc) - round(flux_min / inc));

    // Test both single and double precision.
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    for (int i_type = 0; i_type < 2; ++i_type)
    {
        // Create a test sky model.
        int status = 0;
        oskar_Sky* sky = oskar_sky_create(
                types[i_type], OSKAR_CPU, num_sources, &status
        );
        for (int i = 0; i < num_sources; ++i)
        {
            double flux = inc * i;
            bool in_range = (flux > flux_min && flux <= flux_max);
            double value = in_range ? 1000.0 : 0.0;
            oskar_sky_set_source(
                    sky, i, flux, value, flux, value, value, value,
                    value, value, value, value, value, value, &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }

        // Apply flux filter.
        oskar_sky_filter_by_flux(sky, flux_min, flux_max, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check that there are no sources with fluxes outside the range.
        ASSERT_EQ(expected, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
        for (int i = 0; i < expected; ++i)
        {
            double flux = oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, i);
            EXPECT_LE(flux, flux_max) << "Flux filter failed: i=" << i;
            EXPECT_GT(flux, flux_min) << "Flux filter failed: i=" << i;
            EXPECT_EQ(flux, oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, i));
            EXPECT_EQ(1000.0, oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, i));
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
