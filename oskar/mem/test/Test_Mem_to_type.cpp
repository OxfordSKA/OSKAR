/*
 * Copyright (c) 2013-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "utility/oskar_get_error_string.h"
#include "mem/oskar_mem.h"

static const int size = 100;


TEST(Mem, to_char)
{
    int status = 0, size = 100;
    oskar_Mem* mem = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    char* data = oskar_mem_char(mem);
    ASSERT_EQ(0, oskar_mem_is_single(mem));
    ASSERT_EQ(0, oskar_mem_is_double(mem));
    ASSERT_EQ(0, oskar_mem_is_complex(mem));
    ASSERT_EQ(1, oskar_mem_is_scalar(mem));
    ASSERT_EQ(0, oskar_mem_is_matrix(mem));
    (void) data;
    oskar_mem_free(mem, &status);
}


TEST(Mem, to_int)
{
    // Test happy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(OSKAR_INT, OSKAR_CPU, size, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        int* data = oskar_mem_int(mem, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(0, oskar_mem_is_single(mem));
        ASSERT_EQ(0, oskar_mem_is_double(mem));
        ASSERT_EQ(0, oskar_mem_is_complex(mem));
        ASSERT_EQ(1, oskar_mem_is_scalar(mem));
        ASSERT_EQ(0, oskar_mem_is_matrix(mem));
        (void) data;
        oskar_mem_free(mem, &status);
    }

    // Test unhappy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_SINGLE, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        int* data = oskar_mem_int(mem, &status);
        ASSERT_EQ((int) OSKAR_ERR_TYPE_MISMATCH, status);
        (void) data;
        oskar_mem_free(mem, &status);
    }
}


TEST(Mem, to_float)
{
    // Test happy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_SINGLE, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        float* data = oskar_mem_float(mem, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(1, oskar_mem_is_single(mem));
        ASSERT_EQ(0, oskar_mem_is_double(mem));
        ASSERT_EQ(0, oskar_mem_is_complex(mem));
        ASSERT_EQ(1, oskar_mem_is_real(mem));
        ASSERT_EQ(1, oskar_mem_is_scalar(mem));
        ASSERT_EQ(0, oskar_mem_is_matrix(mem));
        (void) data;
        oskar_mem_free(mem, &status);
    }

    // Test happy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_SINGLE_COMPLEX, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        float* data = oskar_mem_float(mem, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        (void) data;
        oskar_mem_free(mem, &status);
    }

    // Test unhappy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_DOUBLE, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        float* data = oskar_mem_float(mem, &status);
        ASSERT_EQ((int) OSKAR_ERR_TYPE_MISMATCH, status);
        (void) data;
        oskar_mem_free(mem, &status);
    }
}


TEST(Mem, to_float2)
{
    // Test happy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_SINGLE_COMPLEX, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        float2* data = oskar_mem_float2(mem, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(1, oskar_mem_is_single(mem));
        ASSERT_EQ(0, oskar_mem_is_double(mem));
        ASSERT_EQ(1, oskar_mem_is_complex(mem));
        ASSERT_EQ(0, oskar_mem_is_real(mem));
        ASSERT_EQ(1, oskar_mem_is_scalar(mem));
        ASSERT_EQ(0, oskar_mem_is_matrix(mem));
        (void) data;
        oskar_mem_free(mem, &status);
    }

    // Test happy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        float2* data = oskar_mem_float2(mem, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        (void) data;
        oskar_mem_free(mem, &status);
    }

    // Test unhappy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        float2* data = oskar_mem_float2(mem, &status);
        ASSERT_EQ((int) OSKAR_ERR_TYPE_MISMATCH, status);
        (void) data;
        oskar_mem_free(mem, &status);
    }
}


TEST(Mem, to_float4c)
{
    // Test happy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        float4c* data = oskar_mem_float4c(mem, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(1, oskar_mem_is_single(mem));
        ASSERT_EQ(0, oskar_mem_is_double(mem));
        ASSERT_EQ(1, oskar_mem_is_complex(mem));
        ASSERT_EQ(0, oskar_mem_is_real(mem));
        ASSERT_EQ(0, oskar_mem_is_scalar(mem));
        ASSERT_EQ(1, oskar_mem_is_matrix(mem));
        (void) data;
        oskar_mem_free(mem, &status);
    }

    // Test unhappy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        float4c* data = oskar_mem_float4c(mem, &status);
        ASSERT_EQ((int) OSKAR_ERR_TYPE_MISMATCH, status);
        (void) data;
        oskar_mem_free(mem, &status);
    }
}


TEST(Mem, to_double)
{
    // Test happy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_DOUBLE, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        double* data = oskar_mem_double(mem, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(0, oskar_mem_is_single(mem));
        ASSERT_EQ(1, oskar_mem_is_double(mem));
        ASSERT_EQ(0, oskar_mem_is_complex(mem));
        ASSERT_EQ(1, oskar_mem_is_real(mem));
        ASSERT_EQ(1, oskar_mem_is_scalar(mem));
        ASSERT_EQ(0, oskar_mem_is_matrix(mem));
        (void) data;
        oskar_mem_free(mem, &status);
    }

    // Test unhappy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_SINGLE, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        double* data = oskar_mem_double(mem, &status);
        ASSERT_EQ((int) OSKAR_ERR_TYPE_MISMATCH, status);
        (void) data;
        oskar_mem_free(mem, &status);
    }
}


TEST(Mem, to_double2)
{
    // Test happy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        double2* data = oskar_mem_double2(mem, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(0, oskar_mem_is_single(mem));
        ASSERT_EQ(1, oskar_mem_is_double(mem));
        ASSERT_EQ(1, oskar_mem_is_complex(mem));
        ASSERT_EQ(0, oskar_mem_is_real(mem));
        ASSERT_EQ(1, oskar_mem_is_scalar(mem));
        ASSERT_EQ(0, oskar_mem_is_matrix(mem));
        (void) data;
        oskar_mem_free(mem, &status);
    }

    // Test unhappy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_SINGLE_COMPLEX, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        double2* data = oskar_mem_double2(mem, &status);
        ASSERT_EQ((int) OSKAR_ERR_TYPE_MISMATCH, status);
        (void) data;
        oskar_mem_free(mem, &status);
    }
}


TEST(Mem, to_double4c)
{
    // Test happy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        double4c* data = oskar_mem_double4c(mem, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(0, oskar_mem_is_single(mem));
        ASSERT_EQ(1, oskar_mem_is_double(mem));
        ASSERT_EQ(1, oskar_mem_is_complex(mem));
        ASSERT_EQ(0, oskar_mem_is_real(mem));
        ASSERT_EQ(0, oskar_mem_is_scalar(mem));
        ASSERT_EQ(1, oskar_mem_is_matrix(mem));
        (void) data;
        oskar_mem_free(mem, &status);
    }

    // Test unhappy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        double4c* data = oskar_mem_double4c(mem, &status);
        ASSERT_EQ((int) OSKAR_ERR_TYPE_MISMATCH, status);
        (void) data;
        oskar_mem_free(mem, &status);
    }
}


TEST(Mem, to_char_const)
{
    int status = 0;
    oskar_Mem* mem = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_clear_contents(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    const char* data = oskar_mem_char_const(mem);
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ((char) 0, data[i]);
    }
    oskar_mem_free(mem, &status);
}


TEST(Mem, to_int_const)
{
    // Test happy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(OSKAR_INT, OSKAR_CPU, size, &status);
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

    // Test unhappy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_SINGLE, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        const int* data = oskar_mem_int_const(mem, &status);
        ASSERT_EQ((int) OSKAR_ERR_TYPE_MISMATCH, status);
        (void) data;
        oskar_mem_free(mem, &status);
    }
}


TEST(Mem, to_float_const)
{
    // Test happy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_SINGLE, OSKAR_CPU, size, &status
        );
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

    // Test unhappy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_DOUBLE, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        const float* data = oskar_mem_float_const(mem, &status);
        ASSERT_EQ((int) OSKAR_ERR_TYPE_MISMATCH, status);
        (void) data;
        oskar_mem_free(mem, &status);
    }
}


TEST(Mem, to_float2_const)
{
    // Test happy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_SINGLE_COMPLEX, OSKAR_CPU, size, &status
        );
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

    // Test unhappy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        const float2* data = oskar_mem_float2_const(mem, &status);
        ASSERT_EQ((int) OSKAR_ERR_TYPE_MISMATCH, status);
        (void) data;
        oskar_mem_free(mem, &status);
    }
}


TEST(Mem, to_float4c_const)
{
    // Test happy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU, size, &status
        );
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

    // Test unhappy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        const float4c* data = oskar_mem_float4c_const(mem, &status);
        ASSERT_EQ((int) OSKAR_ERR_TYPE_MISMATCH, status);
        (void) data;
        oskar_mem_free(mem, &status);
    }
}


TEST(Mem, to_double_const)
{
    // Test happy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_DOUBLE, OSKAR_CPU, size, &status
        );
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

    // Test unhappy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_SINGLE, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        const double* data = oskar_mem_double_const(mem, &status);
        ASSERT_EQ((int) OSKAR_ERR_TYPE_MISMATCH, status);
        (void) data;
        oskar_mem_free(mem, &status);
    }
}


TEST(Mem, to_double2_const)
{
    // Test happy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, size, &status
        );
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

    // Test unhappy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_SINGLE_COMPLEX, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        const double2* data = oskar_mem_double2_const(mem, &status);
        ASSERT_EQ((int) OSKAR_ERR_TYPE_MISMATCH, status);
        (void) data;
        oskar_mem_free(mem, &status);
    }
}


TEST(Mem, to_double4c_const)
{
    // Test happy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_CPU, size, &status
        );
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

    // Test unhappy path.
    {
        int status = 0;
        oskar_Mem* mem = oskar_mem_create(
                OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU, size, &status
        );
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        const double4c* data = oskar_mem_double4c_const(mem, &status);
        ASSERT_EQ((int) OSKAR_ERR_TYPE_MISMATCH, status);
        (void) data;
        oskar_mem_free(mem, &status);
    }
}
