/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <cstdio>

#include <gtest/gtest.h>

#include "sky/oskar_sky.h"
#include "utility/oskar_device.h"
#include "utility/oskar_get_error_string.h"


TEST(Sky, resize)
{
    int device_loc = 0;
    (void) oskar_device_count(NULL, &device_loc);
    const int locations[] = {OSKAR_CPU, device_loc};

    // Test on both CPU and device.
    for (int i_loc = 0; i_loc < 2; ++i_loc)
    {
        int num_sources = 10;
        int status = 0;
        const int location = locations[i_loc];

        // Create an empty sky model.
        oskar_Sky* sky = oskar_sky_create(OSKAR_DOUBLE, location, 0, &status);

        // Resize.
        oskar_sky_resize(sky, num_sources, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Add some data.
        for (int i = 0; i < num_sources; ++i)
        {
            oskar_sky_set_data(
                    sky, OSKAR_SKY_RA_RAD, 0, i, 1. * i, &status
            );
            oskar_sky_set_data(
                    sky, OSKAR_SKY_DEC_RAD, 0, i, 2. * i, &status
            );
            oskar_sky_set_data(
                    sky, OSKAR_SKY_REF_HZ, 0, i, 100e6 + 1e6 * i, &status
            );
            oskar_sky_set_data(
                    sky, OSKAR_SKY_MAJOR_RAD, 0, i, 3. * i, &status
            );
            oskar_sky_set_data(
                    sky, OSKAR_SKY_MINOR_RAD, 0, i, 4. * i, &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }

        // Grow the sky model.
        num_sources = 100;
        oskar_sky_resize(sky, num_sources, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the existing data values are still correct.
        {
            const oskar_Mem* ra = oskar_sky_column_const(
                    sky, OSKAR_SKY_RA_RAD, 0
            );
            const oskar_Mem* dec = oskar_sky_column_const(
                    sky, OSKAR_SKY_DEC_RAD, 0
            );
            const oskar_Mem* ref = oskar_sky_column_const(
                    sky, OSKAR_SKY_REF_HZ, 0
            );
            const oskar_Mem* maj = oskar_sky_column_const(
                    sky, OSKAR_SKY_MAJOR_RAD, 0
            );
            const oskar_Mem* min = oskar_sky_column_const(
                    sky, OSKAR_SKY_MINOR_RAD, 0
            );
            for (int i = 0; i < num_sources; ++i)
            {
                if (i < 10)
                {
                    ASSERT_DOUBLE_EQ(
                            1. * i, oskar_mem_get_element(ra, i, &status)
                    );
                    ASSERT_DOUBLE_EQ(
                            2. * i, oskar_mem_get_element(dec, i, &status)
                    );
                    ASSERT_DOUBLE_EQ(
                            3. * i, oskar_mem_get_element(maj, i, &status)
                    );
                    ASSERT_DOUBLE_EQ(
                            4. * i, oskar_mem_get_element(min, i, &status)
                    );
                    ASSERT_DOUBLE_EQ(
                            100e6 + 1e6 * i,
                            oskar_mem_get_element(ref, i, &status)
                    );
                }
                else if (location == OSKAR_CPU)
                {
                    // Only check values are zero-initialised on the CPU.
                    ASSERT_DOUBLE_EQ(
                            0., oskar_mem_get_element(ra, i, &status)
                    );
                    ASSERT_DOUBLE_EQ(
                            0., oskar_mem_get_element(dec, i, &status)
                    );
                    ASSERT_DOUBLE_EQ(
                            0., oskar_mem_get_element(maj, i, &status)
                    );
                    ASSERT_DOUBLE_EQ(
                            0., oskar_mem_get_element(min, i, &status)
                    );
                    ASSERT_DOUBLE_EQ(
                            0., oskar_mem_get_element(ref, i, &status)
                    );
                }
            }
        }

        // Set the new values and add an extra column.
        for (int i = 0; i < num_sources; ++i)
        {
            oskar_sky_set_data(
                    sky, OSKAR_SKY_RA_RAD, 0, i, 1.1 * i, &status
            );
            oskar_sky_set_data(
                    sky, OSKAR_SKY_DEC_RAD, 0, i, 2.2 * i, &status
            );
            oskar_sky_set_data(
                    sky, OSKAR_SKY_REF_HZ, 0, i, 100e6 + 1e6 * i, &status
            );
            oskar_sky_set_data(
                    sky, OSKAR_SKY_SPEC_IDX, 0, i, 0.1 * i, &status
            );
            oskar_sky_set_data(
                    sky, OSKAR_SKY_SPEC_IDX, 1, i, 0.2 * i, &status
            );
            oskar_sky_set_data(
                    sky, OSKAR_SKY_MAJOR_RAD, 0, i, 3.3 * i, &status
            );
            oskar_sky_set_data(
                    sky, OSKAR_SKY_MINOR_RAD, 0, i, 4.4 * i, &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }

        // Shrink the sky model.
        num_sources = 50;
        oskar_sky_resize(sky, num_sources, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check the data values are still correct.
        {
            const oskar_Mem* ra = oskar_sky_column_const(
                    sky, OSKAR_SKY_RA_RAD, 0
            );
            const oskar_Mem* dec = oskar_sky_column_const(
                    sky, OSKAR_SKY_DEC_RAD, 0
            );
            const oskar_Mem* ref = oskar_sky_column_const(
                    sky, OSKAR_SKY_REF_HZ, 0
            );
            const oskar_Mem* spx0 = oskar_sky_column_const(
                    sky, OSKAR_SKY_SPEC_IDX, 0
            );
            const oskar_Mem* spx1 = oskar_sky_column_const(
                    sky, OSKAR_SKY_SPEC_IDX, 1
            );
            const oskar_Mem* maj = oskar_sky_column_const(
                    sky, OSKAR_SKY_MAJOR_RAD, 0
            );
            const oskar_Mem* min = oskar_sky_column_const(
                    sky, OSKAR_SKY_MINOR_RAD, 0
            );
            for (int i = 0; i < num_sources; ++i)
            {
                ASSERT_DOUBLE_EQ(
                        1.1 * i, oskar_mem_get_element(ra, i, &status)
                );
                ASSERT_DOUBLE_EQ(
                        2.2 * i, oskar_mem_get_element(dec, i, &status)
                );
                ASSERT_DOUBLE_EQ(
                        3.3 * i, oskar_mem_get_element(maj, i, &status)
                );
                ASSERT_DOUBLE_EQ(
                        4.4 * i, oskar_mem_get_element(min, i, &status)
                );
                ASSERT_DOUBLE_EQ(
                        0.1 * i, oskar_mem_get_element(spx0, i, &status)
                );
                ASSERT_DOUBLE_EQ(
                        0.2 * i, oskar_mem_get_element(spx1, i, &status)
                );
                ASSERT_DOUBLE_EQ(
                        100e6 + 1e6 * i,
                        oskar_mem_get_element(ref, i, &status)
                );
            }
        }

        // Clean up.
        oskar_sky_free(sky, &status);
    }
}
