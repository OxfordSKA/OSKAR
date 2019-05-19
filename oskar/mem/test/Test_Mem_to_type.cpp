/*
 * Copyright (c) 2013-2019, The University of Oxford
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


TEST(Mem, to_char)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, size,
            &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    char* data = oskar_mem_char(mem);
    for (int i = 0; i < size; ++i)
    {
        data[i] = char(i);
    }
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}

TEST(Mem, to_int)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_INT, OSKAR_CPU, size,
            &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    int* data = oskar_mem_int(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < size; ++i)
    {
        data[i] = i;
    }
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}

TEST(Mem, to_float)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, size,
            &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    float* data = oskar_mem_float(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < size; ++i)
    {
        data[i] = float(i);
    }
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}

TEST(Mem, to_float2)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_SINGLE_COMPLEX, OSKAR_CPU,
            size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    float2* data = oskar_mem_float2(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < size; ++i)
    {
        float2 t;
        t.x = 1.0f;
        t.y = 2.0f;
        data[i] = t;
    }
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}

TEST(Mem, to_float4c)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX,
            OSKAR_CPU, size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    float4c* data = oskar_mem_float4c(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < size; ++i)
    {
        float4c t;
        t.a.x = 1.0f;
        t.a.y = 2.0f;
        t.b.x = 3.0f;
        t.b.y = 4.0f;
        t.c.x = 5.0f;
        t.c.y = 6.0f;
        t.d.x = 7.0f;
        t.d.y = 8.0f;
        data[i] = t;
    }
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}

TEST(Mem, to_double)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, size,
            &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    double* data = oskar_mem_double(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < size; ++i)
    {
        data[i] = double(i);
    }
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}

TEST(Mem, to_double2)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_CPU,
            size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    double2* data = oskar_mem_double2(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < size; ++i)
    {
        double2 t;
        t.x = 1.0;
        t.y = 2.0;
        data[i] = t;
    }
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}

TEST(Mem, to_double4c)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_DOUBLE_COMPLEX_MATRIX,
            OSKAR_CPU, size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    double4c* data = oskar_mem_double4c(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < size; ++i)
    {
        double4c t;
        t.a.x = 1.0;
        t.a.y = 2.0;
        t.b.x = 3.0;
        t.b.y = 4.0;
        t.c.x = 5.0;
        t.c.y = 6.0;
        t.d.x = 7.0;
        t.d.y = 8.0;
        data[i] = t;
    }
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}

TEST(Mem, to_const_char)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, size,
            &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_clear_contents(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    const char* data = oskar_mem_char_const(mem);
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ((char)0, data[i]);
    }
    oskar_mem_free(mem, &status);
}

TEST(Mem, to_const_int)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_INT, OSKAR_CPU, size,
            &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_clear_contents(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    const int* data = oskar_mem_int_const(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(0, data[i]);
    }
    oskar_mem_free(mem, &status);
}

TEST(Mem, to_const_float)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, size,
            &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_set_value_real(mem, 2.0, 0, size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    const float* data = oskar_mem_float_const(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < size; ++i)
    {
        ASSERT_FLOAT_EQ(2.0, data[i]);
    }
    oskar_mem_free(mem, &status);
}

TEST(Mem, to_const_float2)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_SINGLE_COMPLEX, OSKAR_CPU,
            size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_set_value_real(mem, 2.0, 0, size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    const float2* data = oskar_mem_float2_const(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < size; ++i)
    {
        ASSERT_FLOAT_EQ(2.0, data[i].x);
        ASSERT_FLOAT_EQ(0.0, data[i].y);
    }
    oskar_mem_free(mem, &status);
}

TEST(Mem, to_const_float4c)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX,
            OSKAR_CPU, size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_set_value_real(mem, 2.0, 0, size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    const float4c* data = oskar_mem_float4c_const(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < size; ++i)
    {
        ASSERT_FLOAT_EQ(2.0, data[i].a.x);
        ASSERT_FLOAT_EQ(0.0, data[i].a.y);
        ASSERT_FLOAT_EQ(0.0, data[i].b.x);
        ASSERT_FLOAT_EQ(0.0, data[i].b.y);
        ASSERT_FLOAT_EQ(0.0, data[i].c.x);
        ASSERT_FLOAT_EQ(0.0, data[i].c.y);
        ASSERT_FLOAT_EQ(2.0, data[i].d.x);
        ASSERT_FLOAT_EQ(0.0, data[i].d.y);
    }
    oskar_mem_free(mem, &status);
}

TEST(Mem, to_const_double)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
            size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_set_value_real(mem, 2.0, 0, size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    const double* data = oskar_mem_double_const(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < size; ++i)
    {
        ASSERT_DOUBLE_EQ(2.0, data[i]);
    }
    oskar_mem_free(mem, &status);
}

TEST(Mem, to_const_double2)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_CPU,
            size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_set_value_real(mem, 2.0, 0, size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    const double2* data = oskar_mem_double2_const(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < size; ++i)
    {
        ASSERT_DOUBLE_EQ(2.0, data[i].x);
        ASSERT_DOUBLE_EQ(0.0, data[i].y);
    }
    oskar_mem_free(mem, &status);
}

TEST(Mem, to_const_double4c)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_DOUBLE_COMPLEX_MATRIX,
            OSKAR_CPU, size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_set_value_real(mem, 2.0, 0, size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    const double4c* data = oskar_mem_double4c_const(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < size; ++i)
    {
        ASSERT_DOUBLE_EQ(2.0, data[i].a.x);
        ASSERT_DOUBLE_EQ(0.0, data[i].a.y);
        ASSERT_DOUBLE_EQ(0.0, data[i].b.x);
        ASSERT_DOUBLE_EQ(0.0, data[i].b.y);
        ASSERT_DOUBLE_EQ(0.0, data[i].c.x);
        ASSERT_DOUBLE_EQ(0.0, data[i].c.y);
        ASSERT_DOUBLE_EQ(2.0, data[i].d.x);
        ASSERT_DOUBLE_EQ(0.0, data[i].d.y);
    }
    oskar_mem_free(mem, &status);
}
