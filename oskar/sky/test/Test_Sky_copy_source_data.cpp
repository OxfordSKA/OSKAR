/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "math/oskar_prefix_sum.h"
#include "sky/oskar_sky.h"
#include "sky/oskar_sky_copy_source_data.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_device.h"


TEST(Sky, copy_source_data)
{
    int status = 0;
    int device_loc = 0;
    (void) oskar_device_count(NULL, &device_loc);
    const int in_size = 321;
    const int locations[] = {OSKAR_CPU, device_loc};
    const int types[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    const double tolerances[] = {5e-5, 1e-14};

    // Test in single and double precision, on CPU and device.
    for (int i_type = 0; i_type < 2; ++i_type)
    {
        for (int i_loc = 0; i_loc < 2; ++i_loc)
        {
            const int type = types[i_type];
            const int location = locations[i_loc];
            const double tol = tolerances[i_type];
            oskar_Sky* in1 = oskar_sky_create(
                    type, location, in_size, &status
            );
            oskar_Sky* out1 = oskar_sky_create(
                    type, location, in_size, &status
            );
            oskar_Mem* mask_cpu = oskar_mem_create(
                    OSKAR_INT, OSKAR_CPU, in_size, &status
            );
            oskar_Mem* indices = oskar_mem_create(
                    OSKAR_INT, location, in_size + 1, &status
            );
            oskar_mem_clear_contents(mask_cpu, &status);
            oskar_mem_clear_contents(indices, &status);
            int* mask_ = oskar_mem_int(mask_cpu, &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);

            // Fill a sky model with a specific pattern.
            int out_size_check = 0;
            for (int i = 0; i < in_size; ++i)
            {
                double ra = 0., dec = 0.;
                double stokes[] = {0., 0., 0., 0.};
                if (i % 2 == 0)
                {
                    ra = i / 2 + 0.0;
                    dec = i / 2 + 0.1;
                    stokes[0] = i / 2 + 0.2;
                    stokes[1] = i / 2 + 0.3;
                    stokes[2] = i / 2 + 0.4;
                    stokes[3] = i / 2 + 0.5;
                    mask_[i] = 1; // Should see these in the output.
                    out_size_check++;
                }
                else
                {
                    ra = i + 10000.;
                    dec = i + 10000.1;
                    stokes[0] = i + 10000.2;
                    stokes[1] = i + 10000.3;
                    stokes[2] = i + 10000.4;
                    stokes[3] = i + 10000.5;
                    mask_[i] = 0; // Should not see these in the output.
                }
                oskar_sky_set_source(in1, i,
                        ra, dec, stokes[0], stokes[1], stokes[2], stokes[3],
                        i + 0.6, i + 0.7, i + 0.8, i + 0.9, i + 1.0, i + 1.1,
                        &status
                );
            }
            // const char* file1 = "temp_test_sky_copy_source_data_in.txt";
            // oskar_sky_save(in, file1, &status);
            // (void) remove(file1);

            // Copy input ready to run test.
            oskar_Sky* in2 = oskar_sky_create_copy(in1, location, &status);

            // Calculate prefix sum (scan) of index array.
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
            oskar_Mem* mask = oskar_mem_create_copy(
                    mask_cpu, location, &status
            );
            oskar_prefix_sum(in_size, mask, indices, &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);

            // Copy source data where the mask is true.
            oskar_sky_copy_source_data(in2, mask, indices, out1, &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);

            // Copy output for checking.
            oskar_Sky* out2 = oskar_sky_create_copy(out1, OSKAR_CPU, &status);

            // Check output is as expected.
            // const char* file2 = "temp_test_sky_copy_source_data_out.txt";
            // oskar_sky_save(out1, file2, &status);
            // (void) remove(file2);
            const int out_size = oskar_sky_int(out2, OSKAR_SKY_NUM_SOURCES);
            ASSERT_EQ(out_size_check, out_size);
            for (int i = 0; i < out_size; ++i)
            {
                ASSERT_NEAR(i + 0.0,
                        oskar_sky_data(out2, OSKAR_SKY_RA_RAD, 0, i), tol
                );
                ASSERT_NEAR(i + 0.1,
                        oskar_sky_data(out2, OSKAR_SKY_DEC_RAD, 0, i), tol
                );
                ASSERT_NEAR(i + 0.2,
                        oskar_sky_data(out2, OSKAR_SKY_I_JY, 0, i), tol
                );
                ASSERT_NEAR(i + 0.3,
                        oskar_sky_data(out2, OSKAR_SKY_Q_JY, 0, i), tol
                );
                ASSERT_NEAR(i + 0.4,
                        oskar_sky_data(out2, OSKAR_SKY_U_JY, 0, i), tol
                );
                ASSERT_NEAR(i + 0.5,
                        oskar_sky_data(out2, OSKAR_SKY_V_JY, 0, i), tol
                );
                ASSERT_NEAR(2 * i + 0.6,
                        oskar_sky_data(out2, OSKAR_SKY_REF_HZ, 0, i), tol
                );
                ASSERT_NEAR(2 * i + 0.7,
                        oskar_sky_data(out2, OSKAR_SKY_SPEC_IDX, 0, i), tol
                );
                ASSERT_NEAR(2 * i + 0.8,
                        oskar_sky_data(out2, OSKAR_SKY_RM_RAD, 0, i), tol
                );
                ASSERT_NEAR(2 * i + 0.9,
                        oskar_sky_data(out2, OSKAR_SKY_MAJOR_RAD, 0, i), tol
                );
                ASSERT_NEAR(2 * i + 1.0,
                        oskar_sky_data(out2, OSKAR_SKY_MINOR_RAD, 0, i), tol
                );
                ASSERT_NEAR(2 * i + 1.1,
                        oskar_sky_data(out2, OSKAR_SKY_PA_RAD, 0, i), tol
                );
            }

            // Clean up.
            oskar_mem_free(indices, &status);
            oskar_mem_free(mask, &status);
            oskar_mem_free(mask_cpu, &status);
            oskar_sky_free(in1, &status);
            oskar_sky_free(in2, &status);
            oskar_sky_free(out1, &status);
            oskar_sky_free(out2, &status);
        }
    }
}
