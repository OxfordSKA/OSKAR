/*
 * Copyright (c) 2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "mem/oskar_mem.h"
#include "utility/oskar_device_count.h"
#include "utility/oskar_get_error_string.h"


TEST(Mem, convert_precision_double_to_float)
{
    int status = 0;
    int num_elements = 100;
    oskar_Mem* in = oskar_mem_create(
            OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_CPU, num_elements, &status
    );
    oskar_mem_random_uniform(in, 1, 2, 3, 4, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_Mem* out = oskar_mem_convert_precision(in, OSKAR_SINGLE, &status);
    ASSERT_EQ((int) OSKAR_SINGLE_COMPLEX_MATRIX, oskar_mem_type(out));
    ASSERT_EQ(num_elements, (int) oskar_mem_length(out));
    double* ptr_in = oskar_mem_double(in, &status);
    float* ptr_out = oskar_mem_float(out, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < 8 * num_elements; ++i)
    {
        EXPECT_NEAR(ptr_in[i], ptr_out[i], 1e-6);
    }
    oskar_mem_free(in, &status);
    oskar_mem_free(out, &status);
}


TEST(Mem, convert_precision_float_to_double)
{
    int status = 0;
    int num_elements = 100;
    oskar_Mem* in = oskar_mem_create(
            OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU, num_elements, &status
    );
    oskar_mem_random_uniform(in, 1, 2, 3, 4, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_Mem* out = oskar_mem_convert_precision(in, OSKAR_DOUBLE, &status);
    ASSERT_EQ((int) OSKAR_DOUBLE_COMPLEX_MATRIX, oskar_mem_type(out));
    ASSERT_EQ(num_elements, (int) oskar_mem_length(out));
    float* ptr_in = oskar_mem_float(in, &status);
    double* ptr_out = oskar_mem_double(out, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < 8 * num_elements; ++i)
    {
        EXPECT_NEAR(ptr_in[i], ptr_out[i], 1e-6);
    }
    oskar_mem_free(in, &status);
    oskar_mem_free(out, &status);
}


TEST(Mem, convert_precision_float_to_double_device)
{
    int status = 0;
    int num_elements = 100;
    int location = 0;
    if (oskar_device_count(NULL, &location) > 0)
    {
        oskar_Mem* in = oskar_mem_create(
                OSKAR_SINGLE_COMPLEX_MATRIX, location, num_elements, &status
        );
        oskar_mem_random_uniform(in, 1, 2, 3, 4, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_Mem* out = oskar_mem_convert_precision(in, OSKAR_DOUBLE, &status);
        ASSERT_EQ((int) OSKAR_DOUBLE_COMPLEX_MATRIX, oskar_mem_type(out));
        ASSERT_EQ(num_elements, (int) oskar_mem_length(out));
        oskar_Mem* in_copy = oskar_mem_create_copy(in, OSKAR_CPU, &status);
        float* ptr_in = oskar_mem_float(in_copy, &status);
        double* ptr_out = oskar_mem_double(out, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        for (int i = 0; i < 8 * num_elements; ++i)
        {
            EXPECT_NEAR(ptr_in[i], ptr_out[i], 1e-6);
        }
        oskar_mem_free(in, &status);
        oskar_mem_free(in_copy, &status);
        oskar_mem_free(out, &status);
    }
}


TEST(Mem, convert_precision_unsupported_types)
{
    int status = 0;
    int num_elements = 100;
    oskar_Mem* in = oskar_mem_create(
            OSKAR_SINGLE_COMPLEX, OSKAR_CPU, num_elements, &status
    );
    oskar_mem_random_uniform(in, 1, 2, 3, 4, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_Mem* out = oskar_mem_convert_precision(in, OSKAR_CHAR, &status);
    ASSERT_EQ(NULL, out);
    ASSERT_EQ((int) OSKAR_ERR_BAD_DATA_TYPE, status);
    oskar_mem_free(in, &status);
    oskar_mem_free(out, &status);
}
