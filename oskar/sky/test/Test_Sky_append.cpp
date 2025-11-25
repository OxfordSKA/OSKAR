/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_device.h"


TEST(Sky, append)
{
    int status = 0;
    int device_loc = 0;
    (void) oskar_device_count(NULL, &device_loc);

    // Create and fill sky model 1.
    int sky1_size = 2;
    oskar_Sky* sky1 = oskar_sky_create(OSKAR_SINGLE, device_loc,
            sky1_size, &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(sky1_size, oskar_sky_int(sky1, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ((int) OSKAR_SINGLE, oskar_sky_int(sky1, OSKAR_SKY_PRECISION));
    ASSERT_EQ(device_loc, oskar_sky_int(sky1, OSKAR_SKY_MEM_LOCATION));
    for (int i = 0; i < sky1_size; ++i)
    {
        const double value = (double) i + 0.1;
        oskar_sky_set_source(sky1, i, value, value,
                value, value, value, value,
                value, value, value, value, value, value, &status
        );
    }
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Create and fill sky model 2.
    int sky2_size = 3;
    oskar_Sky* sky2 = oskar_sky_create(OSKAR_SINGLE, OSKAR_CPU,
            sky2_size, &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < sky2_size; ++i)
    {
        const double value = (double) i + 0.5;
        oskar_sky_set_source(sky2, i, value, value,
                value, value, value, value,
                value, value, value, value, value, value, &status
        );
    }
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Append sky2 to sky1.
    oskar_sky_append(sky1, sky2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Create new copy and check contents.
    oskar_Sky* tmp = oskar_sky_create_copy(sky1, OSKAR_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(sky1_size + sky2_size, oskar_sky_int(tmp, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ(12, oskar_sky_int(tmp, OSKAR_SKY_NUM_COLUMNS));
    for (int i = 0; i < oskar_sky_int(tmp, OSKAR_SKY_NUM_SOURCES); ++i)
    {
        for (int c = 0; c < oskar_sky_int(tmp, OSKAR_SKY_NUM_COLUMNS); ++c)
        {
            oskar_SkyColumn col_type = oskar_sky_column_type(tmp, c);
            int col_attr = oskar_sky_column_attribute(tmp, c);
            float val = (float) oskar_sky_data(tmp, col_type, col_attr, i);
            if (i < sky1_size)
            {
                EXPECT_FLOAT_EQ(float(i + 0.1), val);
            }
            else
            {
                EXPECT_FLOAT_EQ(float(i - sky1_size + 0.5), val);
            }
        }
    }

    // Clean up.
    oskar_sky_free(tmp, &status);
    oskar_sky_free(sky1, &status);
    oskar_sky_free(sky2, &status);
}
