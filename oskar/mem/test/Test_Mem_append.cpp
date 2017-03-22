/*
 * Copyright (c) 2013-2017, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
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
            EXPECT_DOUBLE_EQ(value1, data[i]);
        else
            EXPECT_DOUBLE_EQ(value2, data[i]);
    }

    // Free memory.
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}


#ifdef OSKAR_HAVE_CUDA
TEST(Mem, append_gpu)
{
    int status = 0;
    oskar_Mem *mem, *temp, *mem_temp;

    // Initialise.
    mem = oskar_mem_create(OSKAR_SINGLE, OSKAR_GPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // First append.
    int num_values1 = 10;
    float value1 = 1.0;
    vector<float> data1(num_values1, value1);
    oskar_mem_append_raw(mem, (const void*)&data1[0],
            OSKAR_SINGLE, OSKAR_CPU, num_values1, &status);

    // First check.
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(num_values1, (int)oskar_mem_length(mem));
    ASSERT_EQ((int)OSKAR_GPU, oskar_mem_location(mem));
    ASSERT_EQ((int)OSKAR_SINGLE, oskar_mem_type(mem));
    mem_temp = oskar_mem_create_copy(mem, OSKAR_CPU, &status);
    float* data = oskar_mem_float(mem_temp, &status);
    for (int i = 0; i < num_values1; ++i)
    {
        EXPECT_FLOAT_EQ(value1, data[i]);
    }

    // Second append.
    int num_values2 = 5;
    float value2 = 2.0;
    vector<float> data2(num_values2, value2);
    oskar_mem_append_raw(mem, (const void*)&data2[0],
            OSKAR_SINGLE, OSKAR_CPU, num_values2, &status);

    // Second check.
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(num_values1 + num_values2, (int)oskar_mem_length(mem));
    ASSERT_EQ((int)OSKAR_GPU, oskar_mem_location(mem));
    ASSERT_EQ((int)OSKAR_SINGLE, oskar_mem_type(mem));
    temp = oskar_mem_create_copy(mem, OSKAR_CPU, &status);
    data = oskar_mem_float(temp, &status);
    for (int i = 0; i < (int)oskar_mem_length(mem); ++i)
    {
        if (i < num_values1)
            EXPECT_FLOAT_EQ(value1, data[i]);
        else
            EXPECT_FLOAT_EQ(value2, data[i]);
    }

    // Free memory.
    oskar_mem_free(temp, &status);
    oskar_mem_free(mem, &status);
    oskar_mem_free(mem_temp, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}
#endif

