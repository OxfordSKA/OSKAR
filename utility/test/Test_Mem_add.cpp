/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include <oskar_mem.h>
#include <oskar_get_error_string.h>


TEST(Mem, add_matrix_cpu)
{
    // Use case: Two CPU oskar_Mem matrix types are added together.
    int num_elements = 10, status = 0;
    oskar_Mem *in1, *in2, *out;
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
    oskar_mem_add(out, in1, in2, num_elements, &status);
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
    oskar_Mem *in, *in_out;
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

    oskar_mem_add(in_out, in, in_out, num_elements, &status);
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


TEST(Mem, add_gpu)
{
    // Use Case: memory on the GPU.
    int num_elements = 10, status = 0;
    oskar_Mem *in1, *in2, *out;
    in1 = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_GPU,
            num_elements, &status);
    in2 = oskar_mem_create_copy(in1, OSKAR_GPU, &status);
    out = oskar_mem_create_copy(in1, OSKAR_GPU, &status);
    oskar_mem_add(out, in1, in2, num_elements, &status);
    ASSERT_EQ(0, status);
    oskar_mem_free(in1, &status);
    oskar_mem_free(in2, &status);
    oskar_mem_free(out, &status);
}


TEST(Mem, not_enough_output_elements)
{
    // Use Case: Not enough elements in output array.
    int num_elements = 10, status = 0;
    oskar_Mem *in1, *in2, *out;
    in1 = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU,
            num_elements, &status);
    in2 = oskar_mem_create_copy(in1, OSKAR_CPU, &status);
    out = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU,
            num_elements / 2, &status);
    oskar_mem_add(out, in1, in2, num_elements, &status);
    ASSERT_EQ((int)OSKAR_ERR_DIMENSION_MISMATCH, status);
    status = 0;
    oskar_mem_free(in1, &status);
    oskar_mem_free(in2, &status);
    oskar_mem_free(out, &status);
}

TEST(Mem, add_dimension_mismatch)
{
    // Use Case: Dimension mismatch in arrays being added.
    int num_elements = 10, status = 0;
    oskar_Mem *in1, *in2, *out;
    in1 = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU,
            num_elements, &status);
    in2 = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU,
            num_elements / 2, &status);
    out = oskar_mem_create(OSKAR_SINGLE_COMPLEX_MATRIX, OSKAR_CPU,
            num_elements, &status);
    oskar_mem_add(out, in1, in2, num_elements, &status);
    ASSERT_EQ((int)OSKAR_ERR_DIMENSION_MISMATCH, status);
    status = 0;
    oskar_mem_free(in1, &status);
    oskar_mem_free(in2, &status);
    oskar_mem_free(out, &status);
}
