/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_device.h"


TEST(Sky, copy_contents)
{
    int status = 0;
    const int num_copies = 3, src_size = 20, dst_size = src_size * num_copies;
    oskar_Sky *dst = 0, *src = 0;
    dst = oskar_sky_create(OSKAR_DOUBLE, OSKAR_CPU, dst_size, &status);
    src = oskar_sky_create(OSKAR_DOUBLE, OSKAR_CPU, src_size, &status);

    // Fill a small sky model.
    for (int i = 0; i < src_size; ++i)
    {
        oskar_sky_set_source(src, i,
                i + 0.0, i + 0.1, i + 0.2, i + 0.3, i + 0.4, i + 0.5,
                i + 0.6, i + 0.7, i + 0.8, i + 0.9, i + 1.0, i + 1.1, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    // Copy small sky model into a larger one with an offset.
    for (int i = 0; i < num_copies; ++i)
    {
        oskar_sky_copy_contents(dst, src, i * src_size, 0, src_size, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    // Check contents of the larger sky model.
    const int num_columns = oskar_sky_int(dst, OSKAR_SKY_NUM_COLUMNS);
    ASSERT_EQ(12, num_columns);
    for (int j = 0, s = 0; j < num_copies; ++j)
    {
        for (int i = 0; i < src_size; ++i, ++s)
        {
            for (int c = 0; c < num_columns; ++c)
            {
                oskar_SkyColumn col_type = oskar_sky_column_type(dst, c);
                int col_attr = oskar_sky_column_attribute(dst, c);
                double val = oskar_sky_data(dst, col_type, col_attr, s);
                EXPECT_DOUBLE_EQ(i + 0.1 * c, val);
            }
        }
    }
    oskar_sky_free(src, &status);
    oskar_sky_free(dst, &status);
}
