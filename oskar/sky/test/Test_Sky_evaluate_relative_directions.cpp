/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "math/oskar_cmath.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_device.h"

#define DEG2RAD (M_PI / 180.0)


TEST(Sky, evaluate_relative_directions)
{
    int status = 0;
    int device_loc = 0;
    (void) oskar_device_count(NULL, &device_loc);

    // Construct a simple sky model.
    oskar_Sky* sky1 = oskar_sky_create(OSKAR_DOUBLE, device_loc, 2, &status);
    ASSERT_EQ(2, oskar_sky_int(sky1, OSKAR_SKY_NUM_SOURCES));
    oskar_sky_set_data(sky1, OSKAR_SKY_RA_RAD, 0, 0, 30 * DEG2RAD, &status);
    oskar_sky_set_data(sky1, OSKAR_SKY_RA_RAD, 0, 1, 45 * DEG2RAD, &status);
    oskar_sky_set_data(sky1, OSKAR_SKY_DEC_RAD, 0, 0, 50 * DEG2RAD, &status);
    oskar_sky_set_data(sky1, OSKAR_SKY_DEC_RAD, 0, 1, 60 * DEG2RAD, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Define phase centre.
    const double ra0 = 30.0 * DEG2RAD, dec0 = 55.0 * DEG2RAD;

    // Compute direction cosines.
    oskar_sky_evaluate_relative_directions(sky1, ra0, dec0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Make a copy of the sky model.
    oskar_Sky* sky2 = oskar_sky_create_copy(sky1, OSKAR_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check the data.
    const int num_sources = oskar_sky_int(sky1, OSKAR_SKY_NUM_SOURCES);
    for (int i = 0; i < num_sources; ++i)
    {
        double ra = oskar_sky_data(sky2, OSKAR_SKY_RA_RAD, 0, i);
        double dec = oskar_sky_data(sky2, OSKAR_SKY_DEC_RAD, 0, i);
        double l = sin(ra - ra0) * cos(dec);
        double m = cos(dec0) * sin(dec) - sin(dec0) * cos(dec) * cos(ra - ra0);
        double n = sqrt(1.0 - l * l - m * m);
        EXPECT_DOUBLE_EQ(l, oskar_sky_data(sky2, OSKAR_SKY_SCRATCH_L, 0, i));
        EXPECT_DOUBLE_EQ(m, oskar_sky_data(sky2, OSKAR_SKY_SCRATCH_M, 0, i));
        EXPECT_DOUBLE_EQ(n, oskar_sky_data(sky2, OSKAR_SKY_SCRATCH_N, 0, i));
    }
    oskar_sky_free(sky1, &status);
    oskar_sky_free(sky2, &status);
}
