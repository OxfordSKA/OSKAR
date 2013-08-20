/*
 * Copyright (c) 2013, The University of Oxford
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

#include <oskar_get_error_string.h>
#include <oskar_mem_free.h>
#include <oskar_mem_init.h>
#include <oskar_mem_init_copy.h>
#include <oskar_mem_set_value_real.h>
#include <oskar_vector_types.h>


TEST(Mem, set_value_real_double)
{
    // Double precision real.
    int n = 100, status = 0;
    oskar_Mem mem;
    oskar_mem_init(&mem, OSKAR_DOUBLE, OSKAR_LOCATION_CPU, n, 1, &status);
    oskar_mem_set_value_real(&mem, 4.5, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    for (int i = 0; i < n; ++i)
    {
        EXPECT_DOUBLE_EQ(((double*)mem)[i], 4.5);
    }
    oskar_mem_free(&mem, &status);
}

TEST(Mem, set_value_real_double_complex)
{
    // Double precision complex.
    int n = 100, status = 0;
    oskar_Mem mem, mem2;
    oskar_mem_init(&mem, OSKAR_DOUBLE_COMPLEX,
            OSKAR_LOCATION_GPU, n, 1, &status);
    oskar_mem_set_value_real(&mem, 6.5, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    oskar_mem_init_copy(&mem2, &mem, OSKAR_LOCATION_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        double2 t = ((double2*)mem2)[i];
        EXPECT_DOUBLE_EQ(t.x, 6.5);
        EXPECT_DOUBLE_EQ(t.y, 0.0);
    }
    oskar_mem_free(&mem, &status);
    oskar_mem_free(&mem2, &status);
}

TEST(Mem, set_value_real_double_complex_matrix)
{
    // Double precision complex matrix.
    int n = 100, status = 0;
    oskar_Mem mem, mem2;
    oskar_mem_init(&mem, OSKAR_DOUBLE_COMPLEX_MATRIX,
            OSKAR_LOCATION_GPU, n, 1, &status);
    oskar_mem_set_value_real(&mem, 6.5, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    oskar_mem_init_copy(&mem2, &mem, OSKAR_LOCATION_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        double4c t = ((double4c*)mem2)[i];
        EXPECT_DOUBLE_EQ(t.a.x, 6.5);
        EXPECT_DOUBLE_EQ(t.a.y, 0.0);
        EXPECT_DOUBLE_EQ(t.b.x, 0.0);
        EXPECT_DOUBLE_EQ(t.b.y, 0.0);
        EXPECT_DOUBLE_EQ(t.c.x, 0.0);
        EXPECT_DOUBLE_EQ(t.c.y, 0.0);
        EXPECT_DOUBLE_EQ(t.d.x, 6.5);
        EXPECT_DOUBLE_EQ(t.d.y, 0.0);
    }
    oskar_mem_free(&mem, &status);
    oskar_mem_free(&mem2, &status);
}

TEST(Mem, set_value_real_single)
{
    // Single precision real.
    int n = 100, status = 0;
    oskar_Mem mem;
    oskar_mem_init(&mem, OSKAR_SINGLE, OSKAR_LOCATION_CPU, n, 1, &status);
    oskar_mem_set_value_real(&mem, 4.5, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    for (int i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(((float*)mem)[i], 4.5);
    }
    oskar_mem_free(&mem, &status);
}

TEST(Mem, set_value_real_single_complex)
{
    // Single precision complex.
    int n = 100, status = 0;
    oskar_Mem mem, mem2;
    oskar_mem_init(&mem, OSKAR_SINGLE_COMPLEX,
            OSKAR_LOCATION_GPU, n, 1, &status);
    oskar_mem_set_value_real(&mem, 6.5, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    oskar_mem_init_copy(&mem2, &mem, OSKAR_LOCATION_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        float2 t = ((float2*)mem2)[i];
        EXPECT_FLOAT_EQ(t.x, 6.5);
        EXPECT_FLOAT_EQ(t.y, 0.0);
    }
    oskar_mem_free(&mem, &status);
    oskar_mem_free(&mem2, &status);
}

TEST(Mem, set_value_real_single_complex_matrix)
{
    // Single precision complex matrix.
    int n = 100, status = 0;
    oskar_Mem mem, mem2;
    oskar_mem_init(&mem, OSKAR_SINGLE_COMPLEX_MATRIX,
            OSKAR_LOCATION_GPU, n, 1, &status);
    oskar_mem_set_value_real(&mem, 6.5, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    oskar_mem_init_copy(&mem2, &mem, OSKAR_LOCATION_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        float4c t = ((float4c*)mem2)[i];
        EXPECT_FLOAT_EQ(t.a.x, 6.5);
        EXPECT_FLOAT_EQ(t.a.y, 0.0);
        EXPECT_FLOAT_EQ(t.b.x, 0.0);
        EXPECT_FLOAT_EQ(t.b.y, 0.0);
        EXPECT_FLOAT_EQ(t.c.x, 0.0);
        EXPECT_FLOAT_EQ(t.c.y, 0.0);
        EXPECT_FLOAT_EQ(t.d.x, 6.5);
        EXPECT_FLOAT_EQ(t.d.y, 0.0);
    }
    oskar_mem_free(&mem, &status);
    oskar_mem_free(&mem2, &status);
}
