/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "math/oskar_cmath.h"
#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_device.h"
#include "utility/oskar_timer.h"

#define DEG2RAD (M_PI / 180.0)
#define ARCSEC2RAD (DEG2RAD / 3600.0)


TEST(Sky, load_ascii_all)
{
    // Load an old-format sky file with all columns specified.
    const char* file1 = "temp_test_sources.osm";
    FILE* file = fopen(file1, "w");
    const int num_sources = 10130;
    for (int i = 0; i < num_sources; ++i)
    {
        if (i % 10 == 0) fprintf(file, "# some comment\n");
        (void) fprintf(
                file, "%f %f %f %f %f %f %f %f %f %f %f %f\n",
                i / 10., i / 20.,
                20. * i, 30. * i, 2., 3.,
                1e6 * i, -0.7, 0.5,
                2. * i, 3. * i, 4. * i
        );
    }
    (void) fclose(file);

    // Load the file.
    int status = 0;
    oskar_Timer* timer = oskar_timer_create(OSKAR_TIMER_NATIVE);
    oskar_timer_start(timer);
    oskar_Sky* sky = oskar_sky_load(file1, OSKAR_DOUBLE, &status);
    oskar_timer_pause(timer);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(num_sources, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    printf(
            "Loaded %d sources in %.3f sec\n",
            num_sources, oskar_timer_elapsed(timer)
    );
    oskar_timer_free(timer);

    // Check the data loaded correctly.
    for (int i = 0; i < num_sources; ++i)
    {
        EXPECT_DOUBLE_EQ(
                (i / 10.) * DEG2RAD,
                oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, i)
        );
        EXPECT_DOUBLE_EQ(
                (i / 20.) * DEG2RAD,
                oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, i)
        );
        EXPECT_DOUBLE_EQ(
                20. * i, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, i)
        );
        EXPECT_DOUBLE_EQ(
                30. * i, oskar_sky_data(sky, OSKAR_SKY_Q_JY, 0, i)
        );
        EXPECT_DOUBLE_EQ(
                2., oskar_sky_data(sky, OSKAR_SKY_U_JY, 0, i)
        );
        EXPECT_DOUBLE_EQ(
                3., oskar_sky_data(sky, OSKAR_SKY_V_JY, 0, i)
        );
        EXPECT_DOUBLE_EQ(
                1e6 * i, oskar_sky_data(sky, OSKAR_SKY_REF_HZ, 0, i)
        );
        EXPECT_DOUBLE_EQ(
                -0.7,  oskar_sky_data(sky, OSKAR_SKY_SPEC_IDX, 0, i)
        );
        EXPECT_DOUBLE_EQ(
                0.5, oskar_sky_data(sky, OSKAR_SKY_RM_RAD, 0, i)
        );
        EXPECT_DOUBLE_EQ(
                (2. * i) * ARCSEC2RAD,
                oskar_sky_data(sky, OSKAR_SKY_MAJOR_RAD, 0, i)
        );
        EXPECT_DOUBLE_EQ(
                (3. * i) * ARCSEC2RAD,
                oskar_sky_data(sky, OSKAR_SKY_MINOR_RAD, 0, i)
        );
        EXPECT_DOUBLE_EQ(
                (4. * i) * DEG2RAD,
                oskar_sky_data(sky, OSKAR_SKY_PA_RAD, 0, i)
        );
    }

    // Clean up.
    oskar_sky_free(sky, &status);
    (void) remove(file1);
}


TEST(Sky, load_save_ascii_minimal)
{
    // Load an old-format sky file with just RA, Dec and I specified.
    const char* file1 = "temp_test_sources_minimal.osm";
    const char* file2 = "temp_test_sources_minimal.txt";
    FILE* file = fopen(file1, "w");
    const int num_sources = 521;
    for (int i = 0; i < num_sources; ++i)
    {
        if (i % 10 == 0) fprintf(file, "# some comment\n");
        (void) fprintf(file, "%f, %f, %f\n", i / 10., i / 20., (double) i);
    }
    (void) fclose(file);

    // Load the sky model.
    {
        int status = 0;
        oskar_Sky* sky = oskar_sky_load(file1, OSKAR_DOUBLE, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(num_sources, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));

        // Unspecified columns should not exist.
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_Q_JY, 0));
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_U_JY, 0));
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_V_JY, 0));
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_REF_HZ, 0));
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_SPEC_IDX, 0));
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_RM_RAD, 0));
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_MAJOR_RAD, 0));
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_MINOR_RAD, 0));
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_PA_RAD, 0));

        // Check the data loaded correctly.
        for (int i = 0; i < num_sources; ++i)
        {
            EXPECT_DOUBLE_EQ(
                    (i / 10.) * DEG2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, i)
            );
            EXPECT_DOUBLE_EQ(
                    (i / 20.) * DEG2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, i)
            );
            EXPECT_DOUBLE_EQ(
                    (double) i, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, i)
            );
        }

        // Save the sky model out to check it can be loaded again.
        oskar_sky_save(sky, file2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_sky_free(sky, &status);
    }

    // Load back the sky model that was just saved.
    {
        int status = 0;
        oskar_Sky* sky = oskar_sky_load(file2, OSKAR_DOUBLE, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(num_sources, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));

        // Unspecified columns should not exist (should not have been saved).
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_Q_JY, 0));
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_U_JY, 0));
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_V_JY, 0));
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_REF_HZ, 0));
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_SPEC_IDX, 0));
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_RM_RAD, 0));
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_MAJOR_RAD, 0));
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_MINOR_RAD, 0));
        ASSERT_EQ(NULL, oskar_sky_column_const(sky, OSKAR_SKY_PA_RAD, 0));

        // Check the data loaded correctly.
        for (int i = 0; i < num_sources; ++i)
        {
            EXPECT_DOUBLE_EQ(
                    (i / 10.) * DEG2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_RA_RAD, 0, i)
            );
            EXPECT_DOUBLE_EQ(
                    (i / 20.) * DEG2RAD,
                    oskar_sky_data(sky, OSKAR_SKY_DEC_RAD, 0, i)
            );
            EXPECT_DOUBLE_EQ(
                    (double) i, oskar_sky_data(sky, OSKAR_SKY_I_JY, 0, i)
            );
        }
        oskar_sky_free(sky, &status);
    }

    // Clean up.
    (void) remove(file1);
    (void) remove(file2);
}


TEST(Sky, load_ascii_no_flux)
{
    // Try to load an old-format sky file without enough columns.
    int status = 0;

    // Create a test file to load.
    const char* filename = "temp_test_sources_no_flux.osm";
    FILE* file = fopen(filename, "w");
    const int num_sources = 4;
    for (int i = 0; i < num_sources; ++i)
    {
        if (i % 10 == 0) fprintf(file, "# some comment\n");
        (void) fprintf(file, "%f, %f\n", i / 10., i / 20.);
    }
    (void) fclose(file);

    // Load the sky model.
    // Expect the sky model to be empty.
    oskar_Sky* sky = oskar_sky_load(filename, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));

    // Clean up.
    oskar_sky_free(sky, &status);
    (void) remove(filename);
}


TEST(Sky, load_ascii_zero_flux)
{
    // Load a sky model containing sources with zero flux.
    int status = 0;

    // Create a test file to load.
    const char* filename = "temp_test_sources_zero_flux.osm";
    FILE* file = fopen(filename, "w");
    const int num_sources = 4;
    for (int i = 0; i < num_sources; ++i)
    {
        if (i % 10 == 0) fprintf(file, "# some comment\n");
        (void) fprintf(file, "%f, %f, %f\n", i / 10., i / 20., 0.);
    }
    (void) fclose(file);

    // Load the sky model.
    oskar_Sky* sky = oskar_sky_load(filename, OSKAR_DOUBLE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(num_sources, oskar_sky_int(sky, OSKAR_SKY_NUM_SOURCES));
    const oskar_Mem* stokes_i = oskar_sky_column_const(sky, OSKAR_SKY_I_JY, 0);
    ASSERT_TRUE(stokes_i != NULL);
    for (int i = 0; i < num_sources; ++i)
    {
        EXPECT_DOUBLE_EQ(0, oskar_mem_get_element(stokes_i, i, &status));
    }

    // Clean up.
    oskar_sky_free(sky, &status);
    (void) remove(filename);
}
