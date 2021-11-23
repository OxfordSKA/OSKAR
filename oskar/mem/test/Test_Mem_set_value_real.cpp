/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
    oskar_Mem *mem = 0;
    mem = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, n, &status);
    oskar_mem_set_value_real(mem, 4.5, 0, n, &status);
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
    oskar_Mem *mem = 0;
    mem = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, n, &status);
    oskar_mem_set_value_real(mem, 4.5, 0, n, &status);
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
    oskar_Mem *mem = 0, *mem2 = 0;
    mem = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, location, n, &status);
    oskar_mem_set_value_real(mem, 6.5, 0, n, &status);
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
    oskar_Mem *mem = 0, *mem2 = 0;
    mem = oskar_mem_create(OSKAR_DOUBLE_COMPLEX_MATRIX, location, n, &status);
    oskar_mem_set_value_real(mem, 6.5, 0, n, &status);
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
    oskar_Mem *mem = 0, *mem2 = 0;
    mem = oskar_mem_create(OSKAR_SINGLE_COMPLEX, location, n, &status);
    oskar_mem_set_value_real(mem, 6.5, 0, n, &status);
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
    oskar_Mem *mem = 0, *mem2 = 0;
    mem = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX, location, n, &status);
    oskar_mem_set_value_real(mem, 6.5, 0, n, &status);
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
