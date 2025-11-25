/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"


TEST(Sky, clear_source_flux)
{
    int status = 0;
    const int num_sources = 100;
    oskar_Sky* sky = oskar_sky_create(
            OSKAR_DOUBLE, OSKAR_CPU, num_sources, &status
    );
    oskar_mem_set_value_real(
            oskar_sky_column(sky, OSKAR_SKY_I_JY, 0, &status),
            1.0, 0, num_sources, &status
    );
    oskar_mem_set_value_real(
            oskar_sky_column(sky, OSKAR_SKY_Q_JY, 0, &status),
            2.0, 0, num_sources, &status
    );
    oskar_mem_set_value_real(
            oskar_sky_column(sky, OSKAR_SKY_U_JY, 0, &status),
            3.0, 0, num_sources, &status
    );
    oskar_mem_set_value_real(
            oskar_sky_column(sky, OSKAR_SKY_V_JY, 0, &status),
            4.0, 0, num_sources, &status
    );
    oskar_mem_set_value_real(
            oskar_sky_column(sky, OSKAR_SKY_SCRATCH_I_JY, 0, &status),
            5.0, 0, num_sources, &status
    );
    oskar_mem_set_value_real(
            oskar_sky_column(sky, OSKAR_SKY_SCRATCH_Q_JY, 0, &status),
            6.0, 0, num_sources, &status
    );
    oskar_mem_set_value_real(
            oskar_sky_column(sky, OSKAR_SKY_SCRATCH_U_JY, 0, &status),
            7.0, 0, num_sources, &status
    );
    oskar_mem_set_value_real(
            oskar_sky_column(sky, OSKAR_SKY_SCRATCH_V_JY, 0, &status),
            8.0, 0, num_sources, &status
    );
    // Bonus one...
    oskar_mem_set_value_real(
            oskar_sky_column(sky, OSKAR_SKY_I_JY, 1, &status),
            9.0, 0, num_sources, &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Clear some source values.
    for (int i = 0; i < num_sources; ++i)
    {
        if (i % 10 == 0)
        {
            oskar_sky_clear_source_flux(sky, i, &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
    }

    // Check that it worked.
    for (int i = 0; i < num_sources; ++i)
    {
        for (int c = 0; c < 4; ++c)
        {
            const double val1 = oskar_sky_data(
                    sky, (oskar_SkyColumn) (c + OSKAR_SKY_I_JY), 0, i
            );
            const double val2 = oskar_sky_data(
                    sky, (oskar_SkyColumn) (c + OSKAR_SKY_SCRATCH_I_JY), 0, i
            );
            ASSERT_EQ((i % 10 == 0) ? 0. : c + 1., val1);
            ASSERT_EQ((i % 10 == 0) ? 0. : c + 5., val2);
        }
        const double val = oskar_sky_data(
                sky, OSKAR_SKY_I_JY, 1, i
        );
        ASSERT_EQ((i % 10 == 0) ? 0. : 9., val);
    }

    // Clean up.
    oskar_sky_free(sky, &status);
}
