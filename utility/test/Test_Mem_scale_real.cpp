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
#include <oskar_mem_scale_real.h>
#include <oskar_vector_types.h>


TEST(Mem, scale_real_single)
{
    // Single precision real.
    int n = 100, status = 0;
    oskar_Mem mem_cpu, mem_cpu2, mem_gpu;

    // Initialise.
    oskar_mem_init(&mem_cpu, OSKAR_SINGLE, OSKAR_LOCATION_CPU, n, 1, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        ((float*)(mem_cpu.data))[i] = (float)i;
    }

    // Scale and check contents.
    oskar_mem_scale_real(&mem_cpu, 2.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(2.0f * i, ((float*)(mem_cpu.data))[i]);
    }

    // Copy to GPU and scale again.
    oskar_mem_init_copy(&mem_gpu, &mem_cpu, OSKAR_LOCATION_GPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_scale_real(&mem_gpu, 2.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy back and check contents.
    oskar_mem_init_copy(&mem_cpu2, &mem_gpu, OSKAR_LOCATION_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        EXPECT_FLOAT_EQ(4.0f * i, ((float*)(mem_cpu2.data))[i]);
    }

    // Free memory.
    oskar_mem_free(&mem_cpu, &status);
    oskar_mem_free(&mem_cpu2, &status);
    oskar_mem_free(&mem_gpu, &status);
}


TEST(Mem, scale_real_single_complex)
{
    // Single precision complex.
    int n = 100, status = 0;
    oskar_Mem mem_cpu, mem_cpu2, mem_gpu;

    // Initialise.
    oskar_mem_init(&mem_cpu, OSKAR_SINGLE_COMPLEX,
            OSKAR_LOCATION_CPU, n, 1, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        ((float2*)(mem_cpu.data))[i].x = (float)i;
        ((float2*)(mem_cpu.data))[i].y = (float)i + 0.2f;
    }

    // Scale and check contents.
    oskar_mem_scale_real(&mem_cpu, 2.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        float2 t = ((float2*)(mem_cpu.data))[i];
        EXPECT_FLOAT_EQ(2.0f * ((float)i), t.x);
        EXPECT_FLOAT_EQ(2.0f * ((float)i + 0.2f), t.y);
    }

    // Copy to GPU and scale again.
    oskar_mem_init_copy(&mem_gpu, &mem_cpu, OSKAR_LOCATION_GPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_scale_real(&mem_gpu, 2.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy back and check contents.
    oskar_mem_init_copy(&mem_cpu2, &mem_gpu, OSKAR_LOCATION_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        float2 t = ((float2*)(mem_cpu2.data))[i];
        EXPECT_FLOAT_EQ(4.0f * ((float)i), t.x);
        EXPECT_FLOAT_EQ(4.0f * ((float)i + 0.2f), t.y);
    }

    // Free memory.
    oskar_mem_free(&mem_cpu, &status);
    oskar_mem_free(&mem_cpu2, &status);
    oskar_mem_free(&mem_gpu, &status);
}


TEST(Mem, scale_real_single_complex_matrix)
{
    // Single precision complex matrix.
    int n = 100, status = 0;
    oskar_Mem mem_cpu, mem_cpu2, mem_gpu;

    // Initialise.
    oskar_mem_init(&mem_cpu, OSKAR_SINGLE_COMPLEX_MATRIX,
            OSKAR_LOCATION_CPU, n, 1, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        ((float4c*)(mem_cpu.data))[i].a.x = (float)i;
        ((float4c*)(mem_cpu.data))[i].a.y = (float)i + 0.2f;
        ((float4c*)(mem_cpu.data))[i].b.x = (float)i + 0.4f;
        ((float4c*)(mem_cpu.data))[i].b.y = (float)i + 0.6f;
        ((float4c*)(mem_cpu.data))[i].c.x = (float)i + 0.8f;
        ((float4c*)(mem_cpu.data))[i].c.y = (float)i + 1.0f;
        ((float4c*)(mem_cpu.data))[i].d.x = (float)i + 1.2f;
        ((float4c*)(mem_cpu.data))[i].d.y = (float)i + 1.4f;
    }

    // Scale and check contents.
    oskar_mem_scale_real(&mem_cpu, 2.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        float4c t = ((float4c*)(mem_cpu.data))[i];
        EXPECT_FLOAT_EQ(2.0f * ((float)i), t.a.x);
        EXPECT_FLOAT_EQ(2.0f * ((float)i + 0.2f), t.a.y);
        EXPECT_FLOAT_EQ(2.0f * ((float)i + 0.4f), t.b.x);
        EXPECT_FLOAT_EQ(2.0f * ((float)i + 0.6f), t.b.y);
        EXPECT_FLOAT_EQ(2.0f * ((float)i + 0.8f), t.c.x);
        EXPECT_FLOAT_EQ(2.0f * ((float)i + 1.0f), t.c.y);
        EXPECT_FLOAT_EQ(2.0f * ((float)i + 1.2f), t.d.x);
        EXPECT_FLOAT_EQ(2.0f * ((float)i + 1.4f), t.d.y);
    }

    // Copy to GPU and scale again.
    oskar_mem_init_copy(&mem_gpu, &mem_cpu, OSKAR_LOCATION_GPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_scale_real(&mem_gpu, 2.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy back and check contents.
    oskar_mem_init_copy(&mem_cpu2, &mem_gpu, OSKAR_LOCATION_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        float4c t = ((float4c*)(mem_cpu2.data))[i];
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
    oskar_mem_free(&mem_cpu, &status);
    oskar_mem_free(&mem_cpu2, &status);
    oskar_mem_free(&mem_gpu, &status);
}


TEST(Mem, scale_real_double)
{
    // Double precision real.
    int n = 100, status = 0;
    oskar_Mem mem_cpu, mem_cpu2, mem_gpu;

    // Initialise.
    oskar_mem_init(&mem_cpu, OSKAR_DOUBLE, OSKAR_LOCATION_CPU, n, 1, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        ((double*)(mem_cpu.data))[i] = (double)i;
    }

    // Scale and check contents.
    oskar_mem_scale_real(&mem_cpu, 2.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        EXPECT_DOUBLE_EQ(2.0f * i, ((double*)(mem_cpu.data))[i]);
    }

    // Copy to GPU and scale again.
    oskar_mem_init_copy(&mem_gpu, &mem_cpu, OSKAR_LOCATION_GPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_scale_real(&mem_gpu, 2.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy back and check contents.
    oskar_mem_init_copy(&mem_cpu2, &mem_gpu, OSKAR_LOCATION_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        EXPECT_DOUBLE_EQ(4.0f * i, ((double*)(mem_cpu2.data))[i]);
    }

    // Free memory.
    oskar_mem_free(&mem_cpu, &status);
    oskar_mem_free(&mem_cpu2, &status);
    oskar_mem_free(&mem_gpu, &status);
}


TEST(Mem, scale_real_double_complex)
{
    // Double precision complex.
    int n = 100, status = 0;
    oskar_Mem mem_cpu, mem_cpu2, mem_gpu;

    // Initialise.
    oskar_mem_init(&mem_cpu, OSKAR_DOUBLE_COMPLEX,
            OSKAR_LOCATION_CPU, n, 1, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        ((double2*)(mem_cpu.data))[i].x = (double)i;
        ((double2*)(mem_cpu.data))[i].y = (double)i + 0.2f;
    }

    // Scale and check contents.
    oskar_mem_scale_real(&mem_cpu, 2.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        double2 t = ((double2*)(mem_cpu.data))[i];
        EXPECT_DOUBLE_EQ(2.0f * ((double)i), t.x);
        EXPECT_DOUBLE_EQ(2.0f * ((double)i + 0.2f), t.y);
    }

    // Copy to GPU and scale again.
    oskar_mem_init_copy(&mem_gpu, &mem_cpu, OSKAR_LOCATION_GPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_scale_real(&mem_gpu, 2.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy back and check contents.
    oskar_mem_init_copy(&mem_cpu2, &mem_gpu, OSKAR_LOCATION_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        double2 t = ((double2*)(mem_cpu2.data))[i];
        EXPECT_DOUBLE_EQ(4.0f * ((double)i), t.x);
        EXPECT_DOUBLE_EQ(4.0f * ((double)i + 0.2f), t.y);
    }

    // Free memory.
    oskar_mem_free(&mem_cpu, &status);
    oskar_mem_free(&mem_cpu2, &status);
    oskar_mem_free(&mem_gpu, &status);
}


TEST(Mem, scale_real_double_complex_matrix)
{
    // Double precision complex matrix.
    int n = 100, status = 0;
    oskar_Mem mem_cpu, mem_cpu2, mem_gpu;

    // Initialise.
    oskar_mem_init(&mem_cpu, OSKAR_DOUBLE_COMPLEX_MATRIX,
            OSKAR_LOCATION_CPU, n, 1, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        ((double4c*)(mem_cpu.data))[i].a.x = (double)i;
        ((double4c*)(mem_cpu.data))[i].a.y = (double)i + 0.2f;
        ((double4c*)(mem_cpu.data))[i].b.x = (double)i + 0.4f;
        ((double4c*)(mem_cpu.data))[i].b.y = (double)i + 0.6f;
        ((double4c*)(mem_cpu.data))[i].c.x = (double)i + 0.8f;
        ((double4c*)(mem_cpu.data))[i].c.y = (double)i + 1.0f;
        ((double4c*)(mem_cpu.data))[i].d.x = (double)i + 1.2f;
        ((double4c*)(mem_cpu.data))[i].d.y = (double)i + 1.4f;
    }

    // Scale and check contents.
    oskar_mem_scale_real(&mem_cpu, 2.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        double4c t = ((double4c*)(mem_cpu.data))[i];
        EXPECT_DOUBLE_EQ(2.0f * ((double)i), t.a.x);
        EXPECT_DOUBLE_EQ(2.0f * ((double)i + 0.2f), t.a.y);
        EXPECT_DOUBLE_EQ(2.0f * ((double)i + 0.4f), t.b.x);
        EXPECT_DOUBLE_EQ(2.0f * ((double)i + 0.6f), t.b.y);
        EXPECT_DOUBLE_EQ(2.0f * ((double)i + 0.8f), t.c.x);
        EXPECT_DOUBLE_EQ(2.0f * ((double)i + 1.0f), t.c.y);
        EXPECT_DOUBLE_EQ(2.0f * ((double)i + 1.2f), t.d.x);
        EXPECT_DOUBLE_EQ(2.0f * ((double)i + 1.4f), t.d.y);
    }

    // Copy to GPU and scale again.
    oskar_mem_init_copy(&mem_gpu, &mem_cpu, OSKAR_LOCATION_GPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_scale_real(&mem_gpu, 2.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy back and check contents.
    oskar_mem_init_copy(&mem_cpu2, &mem_gpu, OSKAR_LOCATION_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n; ++i)
    {
        double4c t = ((double4c*)(mem_cpu2.data))[i];
        EXPECT_DOUBLE_EQ(4.0f * ((double)i), t.a.x);
        EXPECT_DOUBLE_EQ(4.0f * ((double)i + 0.2f), t.a.y);
        EXPECT_DOUBLE_EQ(4.0f * ((double)i + 0.4f), t.b.x);
        EXPECT_DOUBLE_EQ(4.0f * ((double)i + 0.6f), t.b.y);
        EXPECT_DOUBLE_EQ(4.0f * ((double)i + 0.8f), t.c.x);
        EXPECT_DOUBLE_EQ(4.0f * ((double)i + 1.0f), t.c.y);
        EXPECT_DOUBLE_EQ(4.0f * ((double)i + 1.2f), t.d.x);
        EXPECT_DOUBLE_EQ(4.0f * ((double)i + 1.4f), t.d.y);
    }

    // Free memory.
    oskar_mem_free(&mem_cpu, &status);
    oskar_mem_free(&mem_cpu2, &status);
    oskar_mem_free(&mem_gpu, &status);
}


