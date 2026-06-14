/*
 * Copyright (c) 2025-2026, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "math/oskar_prefix_sum.h"
#include "sky/oskar_sky.h"
#include "sky/oskar_sky_copy_source_data.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_device.h"
#include "utility/oskar_timer.h"


TEST(Sky, copy_source_data)
{
    int status = 0;
    int device_loc = 0;
    (void) oskar_device_count(NULL, &device_loc);
    const int in_size = 10000;
    const int num_channels = 50;
    const int locations[] = {OSKAR_CPU, device_loc};
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    const double tolerances[] = {5e-5, 1e-14};
    oskar_Mem* mask = oskar_mem_create(OSKAR_INT, OSKAR_CPU, in_size, &status);
    int* mask_ = oskar_mem_int(mask, &status);

    // Test in single and double precision, on CPU and device.
    for (int i_type = 0; i_type < 2; ++i_type)
    {
        const int type = types[i_type];
        oskar_Sky* in = oskar_sky_create(type, OSKAR_CPU, in_size, &status);
        oskar_mem_clear_contents(mask, &status);

        // Fill a sky model with a specific pattern.
        int out_size_check = 0;
        for (int i = 0; i < in_size; ++i)
        {
            if (i % 2 == 0)
            {
                oskar_sky_set_data(
                        in, OSKAR_SKY_RA_RAD, 0, i, i / 2 + 0.0, &status
                );
                oskar_sky_set_data(
                        in, OSKAR_SKY_DEC_RAD, 0, i, i / 2 + 0.1, &status
                );
                for (int c = 0; c < num_channels && c < i; ++c)
                {
                    oskar_sky_set_data(
                            in, OSKAR_SKY_I_JY, c, i, c + i / 2 + 0.2, &status
                    );
                    oskar_sky_set_data(
                            in, OSKAR_SKY_REF_HZ, c, i, c + i + 0.6, &status
                    );
                }
                oskar_sky_set_data(
                        in, OSKAR_SKY_Q_JY, 0, i, i / 2 + 0.3, &status
                );
                oskar_sky_set_data(
                        in, OSKAR_SKY_U_JY, 0, i, i / 2 + 0.4, &status
                );
                oskar_sky_set_data(
                        in, OSKAR_SKY_V_JY, 0, i, i / 2 + 0.5, &status
                );
                mask_[i] = 1; // Should see these in the output.
                out_size_check++;
            }
            else
            {
                oskar_sky_set_data(
                        in, OSKAR_SKY_RA_RAD, 0, i, i + 10000., &status
                );
                oskar_sky_set_data(
                        in, OSKAR_SKY_DEC_RAD, 0, i, i + 10000.1, &status
                );
                oskar_sky_set_data(
                        in, OSKAR_SKY_I_JY, 0, i, i + 10000.2, &status
                );
                oskar_sky_set_data(
                        in, OSKAR_SKY_Q_JY, 0, i, i + 10000.3, &status
                );
                oskar_sky_set_data(
                        in, OSKAR_SKY_U_JY, 0, i, i + 10000.4, &status
                );
                oskar_sky_set_data(
                        in, OSKAR_SKY_V_JY, 0, i, i + 10000.5, &status
                );
                oskar_sky_set_data(
                        in, OSKAR_SKY_REF_HZ, 0, i, i + 0.6, &status
                );
                mask_[i] = 0; // Should not see these in the output.
            }
            oskar_sky_set_data(in, OSKAR_SKY_SPEC_IDX, 0, i, i + 0.7, &status);
            oskar_sky_set_data(in, OSKAR_SKY_RM_RAD, 0, i, i + 0.8, &status);
            oskar_sky_set_data(in, OSKAR_SKY_MAJOR_RAD, 0, i, i + 0.9, &status);
            oskar_sky_set_data(in, OSKAR_SKY_MINOR_RAD, 0, i, i + 1.0, &status);
            oskar_sky_set_data(in, OSKAR_SKY_PA_RAD, 0, i, i + 1.1, &status);
        }
        oskar_sky_sort_columns(in, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        // const char* file1 = "temp_test_sky_copy_source_data_in.txt";
        // oskar_sky_save_named_columns(in, file1, 0, 1, 1, 0, 0, 0, &status);
        // (void) remove(file1);

        for (int i_loc = 0; i_loc < 2; ++i_loc)
        {
            const int location = locations[i_loc];
            const double tol = tolerances[i_type];
            double expected = 0;
            oskar_Timer* timer = oskar_timer_create(location);
            oskar_Sky* out1 = oskar_sky_create(
                    type, location, in_size, &status
            );
            oskar_Mem* indices = oskar_mem_create(
                    OSKAR_INT, location, in_size + 1, &status
            );
            oskar_mem_clear_contents(indices, &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);

            // Copy input ready to run test.
            oskar_Sky* in_copy = oskar_sky_create_copy(in, location, &status);

            // Calculate prefix sum (scan) of index array.
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            oskar_Mem* mask_copy = oskar_mem_create_copy(
                    mask, location, &status
            );
            oskar_prefix_sum(in_size, mask_copy, indices, &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);

            // Copy source data where the mask is true.
            oskar_timer_start(timer);
            oskar_sky_copy_source_data(
                    in_copy, mask_copy, indices, out1, &status
            );
            printf("oskar_sky_copy_source_data() took %.3f s\n",
                    oskar_timer_elapsed(timer)
            );
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            oskar_timer_free(timer);

            // Copy output for checking.
            oskar_Sky* out2 = oskar_sky_create_copy(out1, OSKAR_CPU, &status);

            // Check output is as expected.
            // const char* file2 = "temp_test_sky_copy_source_data_out.txt";
            // oskar_sky_save_named_columns(
            //         out1, file2, 0, 1, 1, 0, 0, 0, &status
            // );
            // (void) remove(file2);
            const int out_size = oskar_sky_int(out2, OSKAR_SKY_NUM_SOURCES);
            ASSERT_EQ(out_size_check, out_size);
            for (int i = 0; i < out_size; ++i)
            {
                int num_chan_exp = 2 * i < num_channels ? 2 * i : num_channels;
                expected = i + 0.0;
                ASSERT_NEAR(expected,
                        oskar_sky_data(out2, OSKAR_SKY_RA_RAD, 0, i),
                        tol * expected
                );
                expected = i + 0.1;
                ASSERT_NEAR(expected,
                        oskar_sky_data(out2, OSKAR_SKY_DEC_RAD, 0, i),
                        tol * expected
                );
                ASSERT_EQ(num_chan_exp, oskar_sky_num_valid_columns_of_type(
                        out2, OSKAR_SKY_I_JY, i)
                );
                ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(
                        out2, OSKAR_SKY_Q_JY, i)
                );
                ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(
                        out2, OSKAR_SKY_U_JY, i)
                );
                ASSERT_EQ(1, oskar_sky_num_valid_columns_of_type(
                        out2, OSKAR_SKY_V_JY, i)
                );
                ASSERT_EQ(num_chan_exp, oskar_sky_num_valid_columns_of_type(
                        out2, OSKAR_SKY_REF_HZ, i)
                );
                for (int c = 0; c < num_chan_exp; ++c)
                {
                    expected = c + i + 0.2;
                    ASSERT_NEAR(expected,
                            oskar_sky_data(out2, OSKAR_SKY_I_JY, c, i),
                            tol * expected
                    );
                    expected = c + 2 * i + 0.6;
                    ASSERT_NEAR(expected,
                            oskar_sky_data(out2, OSKAR_SKY_REF_HZ, c, i),
                            tol * expected
                    );
                }
                expected = i + 0.3;
                ASSERT_NEAR(expected,
                        oskar_sky_data(out2, OSKAR_SKY_Q_JY, 0, i),
                        tol * expected
                );
                expected = i + 0.4;
                ASSERT_NEAR(expected,
                        oskar_sky_data(out2, OSKAR_SKY_U_JY, 0, i),
                        tol * expected
                );
                expected = i + 0.5;
                ASSERT_NEAR(expected,
                        oskar_sky_data(out2, OSKAR_SKY_V_JY, 0, i),
                        tol * expected
                );
                expected = 2 * i + 0.7;
                ASSERT_NEAR(expected,
                        oskar_sky_data(out2, OSKAR_SKY_SPEC_IDX, 0, i),
                        tol * expected
                );
                expected = 2 * i + 0.8;
                ASSERT_NEAR(expected,
                        oskar_sky_data(out2, OSKAR_SKY_RM_RAD, 0, i),
                        tol * expected
                );
                expected = 2 * i + 0.9;
                ASSERT_NEAR(expected,
                        oskar_sky_data(out2, OSKAR_SKY_MAJOR_RAD, 0, i),
                        tol * expected
                );
                expected = 2 * i + 1.0;
                ASSERT_NEAR(expected,
                        oskar_sky_data(out2, OSKAR_SKY_MINOR_RAD, 0, i),
                        tol * expected
                );
                expected = 2 * i + 1.1;
                ASSERT_NEAR(expected,
                        oskar_sky_data(out2, OSKAR_SKY_PA_RAD, 0, i),
                        tol * expected
                );
            }
            oskar_mem_free(indices, &status);
            oskar_mem_free(mask_copy, &status);
            oskar_sky_free(in_copy, &status);
            oskar_sky_free(out1, &status);
            oskar_sky_free(out2, &status);
        }
        oskar_sky_free(in, &status);
    }
    oskar_mem_free(mask, &status);
}
