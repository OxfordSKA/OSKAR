/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "mem/oskar_mem.h"
#include "utility/oskar_get_error_string.h"

#ifdef OSKAR_HAVE_CUDA
static const int location = OSKAR_GPU;
#else
static const int location = OSKAR_CPU;
#endif

TEST(Mem, scale_real_single)
{
    // Single precision real.
    int n = 100, status = 0;
    oskar_Mem *mem_cpu = 0, *mem_cpu2 = 0, *temp = 0;

    // Initialise.
    mem_cpu = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, n, &status);
    float* data = oskar_mem_float(mem_cpu, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        data[i] = (float)i;
    }

    // Scale and check contents.
    oskar_mem_scale_real(mem_cpu, 2.0, 0, n, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(2.0f * i, data[i]);
    }

    // Copy to device and scale again.
    temp = oskar_mem_create_copy(mem_cpu, location, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_scale_real(temp, 2.0, 0, n, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy back and check contents.
    mem_cpu2 = oskar_mem_create_copy(temp, OSKAR_CPU, &status);
    data = oskar_mem_float(mem_cpu2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(4.0f * i, data[i]);
    }

    // Free memory.
    oskar_mem_free(mem_cpu, &status);
    oskar_mem_free(mem_cpu2, &status);
    oskar_mem_free(temp, &status);
}


TEST(Mem, scale_real_single_complex)
{
    // Single precision complex.
    int n = 100, status = 0;
    oskar_Mem *mem_cpu = 0, *mem_cpu2 = 0, *temp = 0;

    // Initialise.
    mem_cpu = oskar_mem_create(OSKAR_SINGLE_COMPLEX, OSKAR_CPU, n,
            &status);
    float2* data = oskar_mem_float2(mem_cpu, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        data[i].x = (float)i;
        data[i].y = (float)i + 0.2f;
    }

    // Scale and check contents.
    oskar_mem_scale_real(mem_cpu, 2.0, 0, n, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        float2 t = data[i];
        EXPECT_FLOAT_EQ(2.0f * ((float)i), t.x);
        EXPECT_FLOAT_EQ(2.0f * ((float)i + 0.2f), t.y);
    }

    // Copy to device and scale again.
    temp = oskar_mem_create_copy(mem_cpu, location, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_scale_real(temp, 2.0, 0, n, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy back and check contents.
    mem_cpu2 = oskar_mem_create_copy(temp, OSKAR_CPU, &status);
    data = oskar_mem_float2(mem_cpu2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        float2 t = data[i];
        EXPECT_FLOAT_EQ(4.0f * ((float)i), t.x);
        EXPECT_FLOAT_EQ(4.0f * ((float)i + 0.2f), t.y);
    }

    // Free memory.
    oskar_mem_free(mem_cpu, &status);
    oskar_mem_free(mem_cpu2, &status);
    oskar_mem_free(temp, &status);
}


TEST(Mem, scale_real_single_complex_matrix)
{
    // Single precision complex matrix.
    int n = 100, status = 0;
    oskar_Mem *mem_cpu = 0, *mem_cpu2 = 0, *temp = 0;

    // Initialise.
    mem_cpu = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU,
            n, &status);
    float4c* data = oskar_mem_float4c(mem_cpu, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        data[i].a.x = (float)i;
        data[i].a.y = (float)i + 0.2f;
        data[i].b.x = (float)i + 0.4f;
        data[i].b.y = (float)i + 0.6f;
        data[i].c.x = (float)i + 0.8f;
        data[i].c.y = (float)i + 1.0f;
        data[i].d.x = (float)i + 1.2f;
        data[i].d.y = (float)i + 1.4f;
    }

    // Scale and check contents.
    oskar_mem_scale_real(mem_cpu, 2.0, 0, n, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        float4c t = data[i];
        EXPECT_FLOAT_EQ(2.0f * ((float)i), t.a.x);
        EXPECT_FLOAT_EQ(2.0f * ((float)i + 0.2f), t.a.y);
        EXPECT_FLOAT_EQ(2.0f * ((float)i + 0.4f), t.b.x);
        EXPECT_FLOAT_EQ(2.0f * ((float)i + 0.6f), t.b.y);
        EXPECT_FLOAT_EQ(2.0f * ((float)i + 0.8f), t.c.x);
        EXPECT_FLOAT_EQ(2.0f * ((float)i + 1.0f), t.c.y);
        EXPECT_FLOAT_EQ(2.0f * ((float)i + 1.2f), t.d.x);
        EXPECT_FLOAT_EQ(2.0f * ((float)i + 1.4f), t.d.y);
    }

    // Copy to device and scale again.
    temp = oskar_mem_create_copy(mem_cpu, location, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_scale_real(temp, 2.0, 0, n, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy back and check contents.
    mem_cpu2 = oskar_mem_create_copy(temp, OSKAR_CPU, &status);
    data = oskar_mem_float4c(mem_cpu2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        float4c t = data[i];
        EXPECT_FLOAT_EQ(4.0f * ((float)i), t.a.x);
        EXPECT_FLOAT_EQ(4.0f * ((float)i + 0.2f), t.a.y);
        EXPECT_FLOAT_EQ(4.0f * ((float)i + 0.4f), t.b.x);
        EXPECT_FLOAT_EQ(4.0f * ((float)i + 0.6f), t.b.y);
        EXPECT_FLOAT_EQ(4.0f * ((float)i + 0.8f), t.c.x);
        EXPECT_FLOAT_EQ(4.0f * ((float)i + 1.0f), t.c.y);
        EXPECT_FLOAT_EQ(4.0f * ((float)i + 1.2f), t.d.x);
        EXPECT_FLOAT_EQ(4.0f * ((float)i + 1.4f), t.d.y);
    }

    // Free memory.
    oskar_mem_free(mem_cpu, &status);
    oskar_mem_free(mem_cpu2, &status);
    oskar_mem_free(temp, &status);
}


TEST(Mem, scale_real_double)
{
    // Double precision real.
    int n = 100, status = 0;
    oskar_Mem *mem_cpu = 0, *mem_cpu2 = 0, *temp = 0;

    // Initialise.
    mem_cpu = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, n, &status);
    double* data = oskar_mem_double(mem_cpu, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        data[i] = (double)i;
    }

    // Scale and check contents.
    oskar_mem_scale_real(mem_cpu, 2.0, 0, n, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        EXPECT_DOUBLE_EQ(2.0 * i, data[i]);
    }

    // Copy to device and scale again.
    temp = oskar_mem_create_copy(mem_cpu, location, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_scale_real(temp, 2.0, 0, n, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy back and check contents.
    mem_cpu2 = oskar_mem_create_copy(temp, OSKAR_CPU, &status);
    data = oskar_mem_double(mem_cpu2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        EXPECT_DOUBLE_EQ(4.0 * i, data[i]);
    }

    // Free memory.
    oskar_mem_free(mem_cpu, &status);
    oskar_mem_free(mem_cpu2, &status);
    oskar_mem_free(temp, &status);
}


TEST(Mem, scale_real_double_complex)
{
    // Double precision complex.
    int n = 100, status = 0;
    oskar_Mem *mem_cpu = 0, *mem_cpu2 = 0, *temp = 0;

    // Initialise.
    mem_cpu = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, n,
            &status);
    double2* data = oskar_mem_double2(mem_cpu, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        data[i].x = (double)i;
        data[i].y = (double)i + 0.2;
    }

    // Scale and check contents.
    oskar_mem_scale_real(mem_cpu, 2.0, 0, n, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        double2 t = data[i];
        EXPECT_DOUBLE_EQ(2.0 * ((double)i), t.x);
        EXPECT_DOUBLE_EQ(2.0 * ((double)i + 0.2), t.y);
    }

    // Copy to device and scale again.
    temp = oskar_mem_create_copy(mem_cpu, location, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_scale_real(temp, 2.0, 0, n, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy back and check contents.
    mem_cpu2 = oskar_mem_create_copy(temp, OSKAR_CPU, &status);
    data = oskar_mem_double2(mem_cpu2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        double2 t = data[i];
        EXPECT_DOUBLE_EQ(4.0 * ((double)i), t.x);
        EXPECT_DOUBLE_EQ(4.0 * ((double)i + 0.2), t.y);
    }

    // Free memory.
    oskar_mem_free(mem_cpu, &status);
    oskar_mem_free(mem_cpu2, &status);
    oskar_mem_free(temp, &status);
}


TEST(Mem, scale_real_double_complex_matrix)
{
    // Double precision complex matrix.
    int n = 100, status = 0;
    oskar_Mem *mem_cpu = 0, *mem_cpu2 = 0, *temp = 0;

    // Initialise.
    mem_cpu = oskar_mem_create(OSKAR_DOUBLE_COMPLEX_MATRIX,
            OSKAR_CPU, n, &status);
    double4c* data = oskar_mem_double4c(mem_cpu, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        data[i].a.x = (double)i;
        data[i].a.y = (double)i + 0.2;
        data[i].b.x = (double)i + 0.4;
        data[i].b.y = (double)i + 0.6;
        data[i].c.x = (double)i + 0.8;
        data[i].c.y = (double)i + 1.0;
        data[i].d.x = (double)i + 1.2;
        data[i].d.y = (double)i + 1.4;
    }

    // Scale and check contents.
    oskar_mem_scale_real(mem_cpu, 2.0, 0, n, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        double4c t = data[i];
        EXPECT_DOUBLE_EQ(2.0 * ((double)i), t.a.x);
        EXPECT_DOUBLE_EQ(2.0 * ((double)i + 0.2), t.a.y);
        EXPECT_DOUBLE_EQ(2.0 * ((double)i + 0.4), t.b.x);
        EXPECT_DOUBLE_EQ(2.0 * ((double)i + 0.6), t.b.y);
        EXPECT_DOUBLE_EQ(2.0 * ((double)i + 0.8), t.c.x);
        EXPECT_DOUBLE_EQ(2.0 * ((double)i + 1.0), t.c.y);
        EXPECT_DOUBLE_EQ(2.0 * ((double)i + 1.2), t.d.x);
        EXPECT_DOUBLE_EQ(2.0 * ((double)i + 1.4), t.d.y);
    }

    // Copy to device and scale again.
    temp = oskar_mem_create_copy(mem_cpu, location, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_scale_real(temp, 2.0, 0, n, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy back and check contents.
    mem_cpu2 = oskar_mem_create_copy(temp, OSKAR_CPU, &status);
    data = oskar_mem_double4c(mem_cpu2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        double4c t = data[i];
        EXPECT_DOUBLE_EQ(4.0 * ((double)i), t.a.x);
        EXPECT_DOUBLE_EQ(4.0 * ((double)i + 0.2), t.a.y);
        EXPECT_DOUBLE_EQ(4.0 * ((double)i + 0.4), t.b.x);
        EXPECT_DOUBLE_EQ(4.0 * ((double)i + 0.6), t.b.y);
        EXPECT_DOUBLE_EQ(4.0 * ((double)i + 0.8), t.c.x);
        EXPECT_DOUBLE_EQ(4.0 * ((double)i + 1.0), t.c.y);
        EXPECT_DOUBLE_EQ(4.0 * ((double)i + 1.2), t.d.x);
        EXPECT_DOUBLE_EQ(4.0 * ((double)i + 1.4), t.d.y);
    }

    // Free memory.
    oskar_mem_free(mem_cpu, &status);
    oskar_mem_free(mem_cpu2, &status);
    oskar_mem_free(temp, &status);
}
