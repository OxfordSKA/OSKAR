/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <vector>

#include <gtest/gtest.h>

#include "mem/oskar_mem.h"
#include "utility/oskar_get_error_string.h"

using std::vector;


TEST(Mem, append_cpu)
{
    int status = 0;

    // Initialise.
    oskar_Mem* mem = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // First append.
    int num_values1 = 10;
    double value1 = 1.0;
    vector<double> data1(num_values1, value1);
    oskar_mem_append_raw(
            mem, (const void*) &data1[0], OSKAR_DOUBLE,
            OSKAR_CPU, num_values1, &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(num_values1, (int) oskar_mem_length(mem));

    // Second append.
    int num_values2 = 5;
    double value2 = 2.0;
    vector<double> data2(num_values2, value2);
    oskar_mem_append_raw(
            mem, (const void*)&data2[0], OSKAR_DOUBLE,
            OSKAR_CPU, num_values2, &status
    );
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(num_values1 + num_values2, (int) oskar_mem_length(mem));

    // Check values.
    double* ptr = oskar_mem_double(mem, &status);
    for (int i = 0; i < (int) oskar_mem_length(mem); ++i)
    {
        if (i < num_values1)
        {
            EXPECT_DOUBLE_EQ(value1, ptr[i]);
        }
        else
        {
            EXPECT_DOUBLE_EQ(value2, ptr[i]);
        }
    }

    // Clean up.
    oskar_mem_free(mem, &status);
}


TEST(Mem, append_wrong_type)
{
    int status = 0;

    // Initialise.
    const int num_elements_initial = 3;
    oskar_Mem* data = oskar_mem_create(
            OSKAR_SINGLE, OSKAR_CPU, num_elements_initial, &status
    );
    oskar_mem_random_uniform(data, 1, 2, 3, 4, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Attempt to append data of the wrong type.
    int num_values1 = 10;
    double value1 = 1.0;
    vector<double> data1(num_values1, value1);
    oskar_mem_append_raw(
            data, (const void*) &data1[0], OSKAR_DOUBLE,
            OSKAR_CPU, num_values1, &status
    );
    ASSERT_EQ((int) OSKAR_ERR_TYPE_MISMATCH, status);
    ASSERT_EQ(num_elements_initial, (int) oskar_mem_length(data));

    // Clean up.
    oskar_mem_free(data, &status);
}


TEST(Mem, append_wrong_location)
{
    int status = 0;

    // Initialise.
    const int num_elements_initial = 3;
    oskar_Mem* data = oskar_mem_create(
            OSKAR_DOUBLE, OSKAR_CPU, num_elements_initial, &status
    );
    oskar_mem_random_uniform(data, 1, 2, 3, 4, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Attempt to append data with an unsupported location type.
    int num_values1 = 10;
    double value1 = 1.0;
    vector<double> data1(num_values1, value1);
    oskar_mem_append_raw(
            data, (const void*) &data1[0], OSKAR_DOUBLE,
            OSKAR_GPU, num_values1, &status
    );
    ASSERT_EQ((int) OSKAR_ERR_BAD_LOCATION, status);
    ASSERT_EQ(num_elements_initial, (int) oskar_mem_length(data));

    // Clean up.
    oskar_mem_free(data, &status);
}
