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

#include "utility/oskar_get_error_string.h"
#include "mem/oskar_mem.h"
#include "utility/oskar_vector_types.h"

#ifdef OSKAR_HAVE_CUDA
static const int location = OSKAR_GPU;
#else
static const int location = OSKAR_CPU;
#endif


TEST(Mem, set_value_real_double)
{
    // Double precision real.
    int n = 100, status = 0;
    oskar_Mem *mem;
    mem = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, n, &status);
    oskar_mem_set_value_real(mem, 4.5, 0, 0, &status);
    double* v = oskar_mem_double(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    for (int i = 0; i < n; ++i)
    {
        EXPECT_DOUBLE_EQ(v[i], 4.5);
    }
    oskar_mem_free(mem, &status);
}

TEST(Mem, set_value_real_single)
{
    // Single precision real.
    int n = 100, status = 0;
    oskar_Mem *mem;
    mem = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, n, &status);
    oskar_mem_set_value_real(mem, 4.5, 0, 0, &status);
    float* v = oskar_mem_float(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    for (int i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(v[i], 4.5);
    }
    oskar_mem_free(mem, &status);
}

TEST(Mem, set_value_real_double_complex)
{
    // Double precision complex.
    int n = 100, status = 0;
    oskar_Mem *mem, *mem2;
    mem = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, location, n, &status);
    oskar_mem_set_value_real(mem, 6.5, 0, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    mem2 = oskar_mem_create_copy(mem, OSKAR_CPU, &status);
    double2* v = oskar_mem_double2(mem2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        EXPECT_DOUBLE_EQ(v[i].x, 6.5);
        EXPECT_DOUBLE_EQ(v[i].y, 0.0);
    }
    oskar_mem_free(mem, &status);
    oskar_mem_free(mem2, &status);
}

TEST(Mem, set_value_real_double_complex_matrix)
{
    // Double precision complex matrix.
    int n = 100, status = 0;
    oskar_Mem *mem, *mem2;
    mem = oskar_mem_create(OSKAR_DOUBLE_COMPLEX_MATRIX, location, n, &status);
    oskar_mem_set_value_real(mem, 6.5, 0, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    mem2 = oskar_mem_create_copy(mem, OSKAR_CPU, &status);
    double4c* v = oskar_mem_double4c(mem2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        EXPECT_DOUBLE_EQ(v[i].a.x, 6.5);
        EXPECT_DOUBLE_EQ(v[i].a.y, 0.0);
        EXPECT_DOUBLE_EQ(v[i].b.x, 0.0);
        EXPECT_DOUBLE_EQ(v[i].b.y, 0.0);
        EXPECT_DOUBLE_EQ(v[i].c.x, 0.0);
        EXPECT_DOUBLE_EQ(v[i].c.y, 0.0);
        EXPECT_DOUBLE_EQ(v[i].d.x, 6.5);
        EXPECT_DOUBLE_EQ(v[i].d.y, 0.0);
    }
    oskar_mem_free(mem, &status);
    oskar_mem_free(mem2, &status);
}

TEST(Mem, set_value_real_single_complex)
{
    // Single precision complex.
    int n = 100, status = 0;
    oskar_Mem *mem, *mem2;
    mem = oskar_mem_create(OSKAR_SINGLE_COMPLEX, location, n, &status);
    oskar_mem_set_value_real(mem, 6.5, 0, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    mem2 = oskar_mem_create_copy(mem, OSKAR_CPU, &status);
    float2* v = oskar_mem_float2(mem2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(v[i].x, 6.5);
        EXPECT_FLOAT_EQ(v[i].y, 0.0);
    }
    oskar_mem_free(mem, &status);
    oskar_mem_free(mem2, &status);
}

TEST(Mem, set_value_real_single_complex_matrix)
{
    // Single precision complex matrix.
    int n = 100, status = 0;
    oskar_Mem *mem, *mem2;
    mem = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX, location, n, &status);
    oskar_mem_set_value_real(mem, 6.5, 0, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    mem2 = oskar_mem_create_copy(mem, OSKAR_CPU, &status);
    float4c* v = oskar_mem_float4c(mem2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(v[i].a.x, 6.5);
        EXPECT_FLOAT_EQ(v[i].a.y, 0.0);
        EXPECT_FLOAT_EQ(v[i].b.x, 0.0);
        EXPECT_FLOAT_EQ(v[i].b.y, 0.0);
        EXPECT_FLOAT_EQ(v[i].c.x, 0.0);
        EXPECT_FLOAT_EQ(v[i].c.y, 0.0);
        EXPECT_FLOAT_EQ(v[i].d.x, 6.5);
        EXPECT_FLOAT_EQ(v[i].d.y, 0.0);
    }
    oskar_mem_free(mem, &status);
    oskar_mem_free(mem2, &status);
}
