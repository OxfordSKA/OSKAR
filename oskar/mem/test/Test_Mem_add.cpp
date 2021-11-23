/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "mem/oskar_mem.h"
#include "utility/oskar_get_error_string.h"


TEST(Mem, add_matrix_cpu)
{
    // Use case: Two CPU oskar_Mem matrix types are added together.
    int num_elements = 10, status = 0;
    oskar_Mem *in1 = 0, *in2 = 0, *out = 0;
    in1 = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU,
            num_elements, &status);
    in2 = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU,
            num_elements, &status);
    float4c* A = oskar_mem_float4c(in1, &status);
    float4c* B = oskar_mem_float4c(in2, &status);

    for (int i = 0; i < num_elements; ++i)
    {
        A[i].a.x = (float)i + 0.1f;
        A[i].a.y = (float)i + 0.2f;
        A[i].b.x = (float)i + 0.3f;
        A[i].b.y = (float)i + 0.4f;
        A[i].c.x = (float)i + 0.5f;
        A[i].c.y = (float)i + 0.6f;
        A[i].d.x = (float)i + 0.7f;
        A[i].d.y = (float)i + 0.8f;
        B[i].a.x = 1.15f;
        B[i].a.y = 0.15f;
        B[i].b.x = 2.16f;
        B[i].b.y = 0.16f;
        B[i].c.x = 3.17f;
        B[i].c.y = 0.17f;
        B[i].d.x = 4.18f;
        B[i].d.y = 0.18f;
    }

    out = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU,
            num_elements, &status);
    oskar_mem_add(out, in1, in2, 0, 0, 0, num_elements, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    float4c* C = oskar_mem_float4c(out, &status);
    for (int i = 0; i < num_elements; ++i)
    {
        EXPECT_FLOAT_EQ(A[i].a.x + B[i].a.x , C[i].a.x);
        EXPECT_FLOAT_EQ(A[i].a.y + B[i].a.y , C[i].a.y);
        EXPECT_FLOAT_EQ(A[i].b.x + B[i].b.x , C[i].b.x);
        EXPECT_FLOAT_EQ(A[i].b.y + B[i].b.y , C[i].b.y);
        EXPECT_FLOAT_EQ(A[i].c.x + B[i].c.x , C[i].c.x);
        EXPECT_FLOAT_EQ(A[i].c.y + B[i].c.y , C[i].c.y);
        EXPECT_FLOAT_EQ(A[i].d.x + B[i].d.x , C[i].d.x);
        EXPECT_FLOAT_EQ(A[i].d.y + B[i].d.y , C[i].d.y);
    }
    oskar_mem_free(in1, &status);
    oskar_mem_free(in2, &status);
    oskar_mem_free(out, &status);
}


TEST(Mem, add_in_place)
{
    // Use case: In place add.
    int num_elements = 10, status = 0;
    oskar_Mem *in = 0, *in_out = 0;
    in = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU,
            num_elements, &status);
    in_out = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU,
            num_elements, &status);
    float4c* A = oskar_mem_float4c(in, &status);
    float4c* B = oskar_mem_float4c(in_out, &status);

    for (int i = 0; i < num_elements; ++i)
    {
        A[i].a.x = (float)i + 0.1f;
        A[i].a.y = (float)i + 0.2f;
        A[i].b.x = (float)i + 0.3f;
        A[i].b.y = (float)i + 0.4f;
        A[i].c.x = (float)i + 0.5f;
        A[i].c.y = (float)i + 0.6f;
        A[i].d.x = (float)i + 0.7f;
        A[i].d.y = (float)i + 0.8f;
        EXPECT_FLOAT_EQ(0.0f, B[i].a.x);
        EXPECT_FLOAT_EQ(0.0f, B[i].a.y);
        EXPECT_FLOAT_EQ(0.0f, B[i].b.x);
        EXPECT_FLOAT_EQ(0.0f, B[i].b.y);
        EXPECT_FLOAT_EQ(0.0f, B[i].c.x);
        EXPECT_FLOAT_EQ(0.0f, B[i].c.y);
        EXPECT_FLOAT_EQ(0.0f, B[i].d.x);
        EXPECT_FLOAT_EQ(0.0f, B[i].d.y);
    }

    oskar_mem_add(in_out, in, in_out, 0, 0, 0, num_elements, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    for (int i = 0; i < num_elements; ++i)
    {
        EXPECT_FLOAT_EQ(A[i].a.x, B[i].a.x);
        EXPECT_FLOAT_EQ(A[i].a.y, B[i].a.y);
        EXPECT_FLOAT_EQ(A[i].b.x, B[i].b.x);
        EXPECT_FLOAT_EQ(A[i].b.y, B[i].b.y);
        EXPECT_FLOAT_EQ(A[i].c.x, B[i].c.x);
        EXPECT_FLOAT_EQ(A[i].c.y, B[i].c.y);
        EXPECT_FLOAT_EQ(A[i].d.x, B[i].d.x);
        EXPECT_FLOAT_EQ(A[i].d.y, B[i].d.y);
    }
    oskar_mem_free(in, &status);
    oskar_mem_free(in_out, &status);
}


#ifdef OSKAR_HAVE_CUDA
TEST(Mem, add_gpu_single)
{
    int num_elements = 445, prec = OSKAR_SINGLE, status = 0;
    oskar_Mem *in1 = 0, *in2 = 0, *in1_cl = 0, *in2_cl = 0, *out = 0, *out_cl = 0;
    in1 = oskar_mem_create(prec, OSKAR_CPU, num_elements, &status);
    in2 = oskar_mem_create(prec, OSKAR_CPU, num_elements, &status);
    float* A = oskar_mem_float(in1, &status);
    float* B = oskar_mem_float(in2, &status);
    for (int i = 0; i < num_elements; ++i)
    {
        A[i] = (float) i + 0.2f;
        B[i] = (float) (2*i) + 0.4f;
    }
    in1_cl = oskar_mem_create_copy(in1, OSKAR_GPU, &status);
    in2_cl = oskar_mem_create_copy(in2, OSKAR_GPU, &status);
    out_cl = oskar_mem_create(prec, OSKAR_GPU, num_elements, &status);
    ASSERT_EQ(0, status);
    oskar_mem_add(out_cl, in1_cl, in2_cl, 0, 0, 0, num_elements, &status);
    ASSERT_EQ(0, status);
    out = oskar_mem_create_copy(out_cl, OSKAR_CPU, &status);
    float* C = oskar_mem_float(out, &status);
    ASSERT_EQ(0, status);
    for (int i = 0; i < num_elements; ++i)
    {
        EXPECT_FLOAT_EQ(A[i] + B[i], C[i]);
    }
    oskar_mem_free(in1, &status);
    oskar_mem_free(in1_cl, &status);
    oskar_mem_free(in2, &status);
    oskar_mem_free(in2_cl, &status);
    oskar_mem_free(out, &status);
    oskar_mem_free(out_cl, &status);
}
#endif

#ifdef OSKAR_HAVE_OPENCL
TEST(Mem, add_cl_single)
{
    int num_elements = 445, prec = OSKAR_SINGLE, status = 0;
    oskar_Mem *in1 = 0, *in2 = 0, *in1_cl = 0, *in2_cl = 0, *out = 0, *out_cl = 0;
    in1 = oskar_mem_create(prec, OSKAR_CPU, num_elements, &status);
    in2 = oskar_mem_create(prec, OSKAR_CPU, num_elements, &status);
    float* A = oskar_mem_float(in1, &status);
    float* B = oskar_mem_float(in2, &status);
    for (int i = 0; i < num_elements; ++i)
    {
        A[i] = (float) i + 0.2f;
        B[i] = (float) (2*i) + 0.4f;
    }
    in1_cl = oskar_mem_create_copy(in1, OSKAR_CL, &status);
    in2_cl = oskar_mem_create_copy(in2, OSKAR_CL, &status);
    out_cl = oskar_mem_create(prec, OSKAR_CL, num_elements, &status);
    ASSERT_EQ(0, status);
    oskar_mem_add(out_cl, in1_cl, in2_cl, 0, 0, 0, num_elements, &status);
    ASSERT_EQ(0, status);
    out = oskar_mem_create_copy(out_cl, OSKAR_CPU, &status);
    float* C = oskar_mem_float(out, &status);
    ASSERT_EQ(0, status);
    for (int i = 0; i < num_elements; ++i)
    {
        EXPECT_FLOAT_EQ(A[i] + B[i], C[i]);
    }
    oskar_mem_free(in1, &status);
    oskar_mem_free(in1_cl, &status);
    oskar_mem_free(in2, &status);
    oskar_mem_free(in2_cl, &status);
    oskar_mem_free(out, &status);
    oskar_mem_free(out_cl, &status);
}

TEST(Mem, add_cl_double)
{
    int num_elements = 445, prec = OSKAR_DOUBLE, status = 0;
    oskar_Mem *in1 = 0, *in2 = 0, *in1_cl = 0, *in2_cl = 0, *out = 0, *out_cl = 0;
    in1 = oskar_mem_create(prec, OSKAR_CPU, num_elements, &status);
    in2 = oskar_mem_create(prec, OSKAR_CPU, num_elements, &status);
    double* A = oskar_mem_double(in1, &status);
    double* B = oskar_mem_double(in2, &status);
    for (int i = 0; i < num_elements; ++i)
    {
        A[i] = (double) i + 0.2f;
        B[i] = (double) (2*i) + 0.4f;
    }
    in1_cl = oskar_mem_create_copy(in1, OSKAR_CL, &status);
    in2_cl = oskar_mem_create_copy(in2, OSKAR_CL, &status);
    out_cl = oskar_mem_create(prec, OSKAR_CL, num_elements, &status);
    ASSERT_EQ(0, status);
    oskar_mem_add(out_cl, in1_cl, in2_cl, 0, 0, 0, num_elements, &status);
    ASSERT_EQ(0, status);
    out = oskar_mem_create_copy(out_cl, OSKAR_CPU, &status);
    double* C = oskar_mem_double(out, &status);
    ASSERT_EQ(0, status);
    for (int i = 0; i < num_elements; ++i)
    {
        EXPECT_DOUBLE_EQ(A[i] + B[i], C[i]);
    }
    oskar_mem_free(in1, &status);
    oskar_mem_free(in1_cl, &status);
    oskar_mem_free(in2, &status);
    oskar_mem_free(in2_cl, &status);
    oskar_mem_free(out, &status);
    oskar_mem_free(out_cl, &status);
}
#endif
