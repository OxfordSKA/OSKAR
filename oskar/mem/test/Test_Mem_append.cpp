/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "mem/oskar_mem.h"
#include "utility/oskar_get_error_string.h"

#include <vector>

using std::vector;

TEST(Mem, append_cpu)
{
    int status = 0;

    // Initialise.
    oskar_Mem* mem = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 0,
            &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // First append.
    int num_values1 = 10;
    double value1 = 1.0;
    vector<double> data1(num_values1, value1);
    oskar_mem_append_raw(mem, (const void*)&data1[0], OSKAR_DOUBLE,
            OSKAR_CPU, num_values1, &status);

    // First check.
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(num_values1, (int)oskar_mem_length(mem));
    ASSERT_EQ((int)OSKAR_CPU, oskar_mem_location(mem));
    ASSERT_EQ((int)OSKAR_DOUBLE, oskar_mem_type(mem));
    double* data = oskar_mem_double(mem, &status);
    for (int i = 0; i < num_values1; ++i)
    {
        EXPECT_DOUBLE_EQ(value1, data[i]);
    }

    // Second append.
    int num_values2 = 5;
    double value2 = 2.0;
    vector<double> data2(num_values2, value2);
    oskar_mem_append_raw(mem, (const void*)&data2[0], OSKAR_DOUBLE,
            OSKAR_CPU, num_values2, &status);

    // Second check.
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(num_values1 + num_values2, (int)oskar_mem_length(mem));
    ASSERT_EQ((int)OSKAR_CPU, oskar_mem_location(mem));
    ASSERT_EQ((int)OSKAR_DOUBLE, oskar_mem_type(mem));
    data = oskar_mem_double(mem, &status);
    for (int i = 0; i < (int)oskar_mem_length(mem); ++i)
    {
        if (i < num_values1)
        {
            EXPECT_DOUBLE_EQ(value1, data[i]);
        }
        else
        {
            EXPECT_DOUBLE_EQ(value2, data[i]);
        }
    }

    // Free memory.
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}
