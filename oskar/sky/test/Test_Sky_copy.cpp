/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_device.h"


TEST(Sky, copy)
{
    int status = 0;
    int device_loc = 0;
    (void) oskar_device_count(NULL, &device_loc);

    // Create and fill sky model 1.
    const int sky1_size = 50000;
    oskar_Sky* sky1 = oskar_sky_create(
            OSKAR_SINGLE, OSKAR_CPU, sky1_size, &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(sky1_size, oskar_sky_int(sky1, OSKAR_SKY_NUM_SOURCES));
    ASSERT_EQ((int) OSKAR_SINGLE, oskar_sky_int(sky1, OSKAR_SKY_PRECISION));
    ASSERT_EQ((int) OSKAR_CPU, oskar_sky_int(sky1, OSKAR_SKY_MEM_LOCATION));
    for (int i = 0; i < sky1_size; ++i)
    {
        oskar_sky_set_source(sky1, i,
                i + 0.0, i + 0.1, i + 0.2, i + 0.3, i + 0.4, i + 0.5,
                i + 0.6, i + 0.7, i + 0.8, i + 0.9, i + 1.0, i + 1.1, &status
        );
    }
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Create another sky model.
    const int sky2_size = sky1_size;
    oskar_Sky* sky2 = oskar_sky_create(
            OSKAR_SINGLE, device_loc, sky2_size, &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Try to set a source at a non-existent index (this should fail).
    oskar_sky_set_source(sky2, sky2_size, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            200.0e6, -0.7, 2.0, 0.0, 0.0, 0.0, &status
    );
    ASSERT_EQ((int) OSKAR_ERR_OUT_OF_RANGE, status);
    status = 0;

    // Copy sky model 1 into 2.
    oskar_sky_copy(sky2, sky1, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Create new copy and check contents.
    oskar_Sky* tmp = oskar_sky_create_copy(sky2, OSKAR_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(sky1_size, oskar_sky_int(tmp, OSKAR_SKY_NUM_SOURCES));
    const int num_columns = oskar_sky_int(tmp, OSKAR_SKY_NUM_COLUMNS);
    ASSERT_EQ(12, num_columns);
    for (int i = 0; i < sky1_size; ++i)
    {
        for (int c = 0; c < num_columns; ++c)
        {
            oskar_SkyColumn col_type = oskar_sky_column_type(tmp, c);
            int col_attr = oskar_sky_column_attribute(tmp, c);
            float val = (float) oskar_sky_data(tmp, col_type, col_attr, i);
            EXPECT_FLOAT_EQ(float(i + 0.1 * c), val);
        }
    }

    // Clean up.
    oskar_sky_free(tmp, &status);
    oskar_sky_free(sky1, &status);
    oskar_sky_free(sky2, &status);
}
