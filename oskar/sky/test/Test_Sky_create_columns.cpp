/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_device.h"


TEST(Sky, create_columns)
{
    int status = 0;
    int device_loc = 0;
    (void) oskar_device_count(NULL, &device_loc);

    // Create and fill sky model 1.
    const int sky1_size = 16384;
    oskar_Sky* sky1 = oskar_sky_create(
            OSKAR_DOUBLE, OSKAR_CPU, sky1_size, &status
    );
    for (int i = 0; i < sky1_size; ++i)
    {
        // Create 4 regular columns.
        oskar_sky_set_data(sky1, OSKAR_SKY_RA_RAD, 0, i, i * 1., &status);
        oskar_sky_set_data(sky1, OSKAR_SKY_DEC_RAD, 0, i, i * 2., &status);
        oskar_sky_set_data(sky1, OSKAR_SKY_I_JY, 0, i, i * 3., &status);
        oskar_sky_set_data(sky1, OSKAR_SKY_REF_HZ, 0, i, i * 1e3, &status);

        // Include a couple of scratch columns.
        oskar_sky_set_data(
                sky1, OSKAR_SKY_SCRATCH_EXT_A, 0, i, i * 4., &status
        );
        oskar_sky_set_data(
                sky1, OSKAR_SKY_SCRATCH_EXT_C, 0, i, i * 5., &status
        );
    }
    ASSERT_EQ(6, oskar_sky_int(sky1, OSKAR_SKY_NUM_COLUMNS));

    // Create an empty sky model 2.
    oskar_Sky* sky2 = oskar_sky_create(
            OSKAR_DOUBLE, device_loc, sky1_size, &status
    );
    ASSERT_EQ(0, oskar_sky_int(sky2, OSKAR_SKY_NUM_COLUMNS));
    ASSERT_EQ(sky1_size, oskar_sky_int(sky2, OSKAR_SKY_NUM_SOURCES));
    const int num_columns_scratch = (
            OSKAR_SKY_SCRATCH_END - OSKAR_SKY_SCRATCH_START
    );

    // Create columns in sky model 2.
    oskar_sky_create_columns(sky2, sky1, &status);
    const int num_columns = oskar_sky_int(sky2, OSKAR_SKY_NUM_COLUMNS);
    ASSERT_EQ(4 + num_columns_scratch, num_columns);

    // Check that they actually exist.
    ASSERT_EQ(OSKAR_SKY_RA_RAD, oskar_sky_column_type(sky2, 0));
    ASSERT_EQ(OSKAR_SKY_DEC_RAD, oskar_sky_column_type(sky2, 1));
    ASSERT_EQ(OSKAR_SKY_I_JY, oskar_sky_column_type(sky2, 2));
    ASSERT_EQ(OSKAR_SKY_REF_HZ, oskar_sky_column_type(sky2, 3));
    ASSERT_EQ(OSKAR_SKY_SCRATCH_EXT_A, oskar_sky_column_type(sky2, 4));
    ASSERT_EQ(OSKAR_SKY_SCRATCH_EXT_C, oskar_sky_column_type(sky2, 5));
    // The ones below should have been added.
    ASSERT_EQ(OSKAR_SKY_SCRATCH_EXT_B, oskar_sky_column_type(sky2, 6));
    ASSERT_EQ(OSKAR_SKY_SCRATCH_L, oskar_sky_column_type(sky2, 7));
    ASSERT_EQ(OSKAR_SKY_SCRATCH_M, oskar_sky_column_type(sky2, 8));
    ASSERT_EQ(OSKAR_SKY_SCRATCH_N, oskar_sky_column_type(sky2, 9));
    ASSERT_EQ(OSKAR_SKY_SCRATCH_I_JY, oskar_sky_column_type(sky2, 10));
    ASSERT_EQ(OSKAR_SKY_SCRATCH_Q_JY, oskar_sky_column_type(sky2, 11));
    ASSERT_EQ(OSKAR_SKY_SCRATCH_U_JY, oskar_sky_column_type(sky2, 12));
    ASSERT_EQ(OSKAR_SKY_SCRATCH_V_JY, oskar_sky_column_type(sky2, 13));

    // Check column pointers are not NULL.
    ASSERT_TRUE(oskar_sky_column_const(sky2, OSKAR_SKY_RA_RAD, 0) != 0);
    ASSERT_TRUE(oskar_sky_column_const(sky2, OSKAR_SKY_DEC_RAD, 0) != 0);
    ASSERT_TRUE(oskar_sky_column_const(sky2, OSKAR_SKY_I_JY, 0) != 0);
    ASSERT_TRUE(oskar_sky_column_const(sky2, OSKAR_SKY_REF_HZ, 0) != 0);
    ASSERT_TRUE(oskar_sky_column_const(sky2, OSKAR_SKY_SCRATCH_EXT_A, 0) != 0);
    ASSERT_TRUE(oskar_sky_column_const(sky2, OSKAR_SKY_SCRATCH_EXT_B, 0) != 0);
    ASSERT_TRUE(oskar_sky_column_const(sky2, OSKAR_SKY_SCRATCH_EXT_C, 0) != 0);
    ASSERT_TRUE(oskar_sky_column_const(sky2, OSKAR_SKY_SCRATCH_L, 0) != 0);
    ASSERT_TRUE(oskar_sky_column_const(sky2, OSKAR_SKY_SCRATCH_M, 0) != 0);
    ASSERT_TRUE(oskar_sky_column_const(sky2, OSKAR_SKY_SCRATCH_N, 0) != 0);
    ASSERT_TRUE(oskar_sky_column_const(sky2, OSKAR_SKY_SCRATCH_I_JY, 0) != 0);
    ASSERT_TRUE(oskar_sky_column_const(sky2, OSKAR_SKY_SCRATCH_Q_JY, 0) != 0);
    ASSERT_TRUE(oskar_sky_column_const(sky2, OSKAR_SKY_SCRATCH_U_JY, 0) != 0);
    ASSERT_TRUE(oskar_sky_column_const(sky2, OSKAR_SKY_SCRATCH_V_JY, 0) != 0);

    // Clean up.
    oskar_sky_free(sky1, &status);
    oskar_sky_free(sky2, &status);
}
