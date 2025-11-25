/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "sky/oskar_sky.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_device.h"


TEST(Sky, append_to_set)
{
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    const double tolerances[] = {5e-5, 1e-14};
    for (int i_type = 0; i_type < 2; ++i_type)
    {
        int status = 0;
        const int type = types[i_type];
        const double tol = tolerances[i_type];

        // Declare an empty sky model set.
        int num_models_in_set = 0;
        oskar_Sky** set = 0;
        const int max_sources_per_model = 5;

        // Create a sky model and append it to an empty set.
        const int size1 = 6;
        oskar_Sky* sky1 = oskar_sky_create(type, OSKAR_CPU, size1, &status);
        for (int i = 0; i < size1; ++i)
        {
            oskar_sky_set_data(
                    sky1, OSKAR_SKY_RA_RAD, 0, i, (double) i, &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
        oskar_sky_append_to_set(
                &num_models_in_set, &set, max_sources_per_model, sky1, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Create another sky model and append it to the current set.
        const int size2 = 7;
        oskar_Sky* sky2 = oskar_sky_create(type, OSKAR_CPU, size2, &status);
        for (int i = 0; i < size2; ++i)
        {
            oskar_sky_set_data(sky2, OSKAR_SKY_RA_RAD, 0, i, i + 0.5, &status);
            oskar_sky_set_data(
                    sky2, OSKAR_SKY_MAJOR_RAD, 0, i, i * 0.75, &status
            );
            oskar_sky_set_data(
                    sky2, OSKAR_SKY_MINOR_RAD, 0, i, i * 0.25, &status
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
        oskar_sky_append_to_set(
                &num_models_in_set, &set, max_sources_per_model, sky2, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Free the individual sky models.
        oskar_sky_free(sky1, &status);
        oskar_sky_free(sky2, &status);

        // Check the number of models in the set.
        // Sky model 1 had 6 sources, and model 2 had 7 sources.
        // The total number of sources is therefore 13,
        // and we should have 3 sky models.
        EXPECT_EQ(3, num_models_in_set);

        // Check the contents of the set.
        for (int i = 0, s = 0; i < num_models_in_set; ++i)
        {
            const int num_sources = oskar_sky_int(set[i], OSKAR_SKY_NUM_SOURCES);
            if (i != num_models_in_set - 1)
            {
                EXPECT_EQ(max_sources_per_model, num_sources);
            }
            else
            {
                // Last one.
                EXPECT_EQ(3, num_sources);
            }
            for (int j = 0; j < num_sources; ++j, ++s)
            {
                double ra = oskar_sky_data(set[i], OSKAR_SKY_RA_RAD, 0, j);
                double maj = oskar_sky_data(set[i], OSKAR_SKY_MAJOR_RAD, 0, j);
                double min = oskar_sky_data(set[i], OSKAR_SKY_MINOR_RAD, 0, j);
                if (s < size1)
                {
                    EXPECT_NEAR((double) s, ra, tol);
                    EXPECT_NEAR(0.0, maj, tol);
                    EXPECT_NEAR(0.0, min, tol);
                }
                else
                {
                    EXPECT_NEAR(1.0 * (s - size1) + 0.5, ra, tol);
                    EXPECT_NEAR((s - size1) * 0.75, maj, tol);
                    EXPECT_NEAR((s - size1) * 0.25, min, tol);
                }
            }
        }

        // Free the array of sky models.
        for (int i = 0; i < num_models_in_set; ++i)
        {
            oskar_sky_free(set[i], &status);
        }
        free(set);
    }
}
