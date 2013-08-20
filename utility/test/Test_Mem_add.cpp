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
#include <oskar_mem_add.h>
#include <oskar_mem_free.h>
#include <oskar_mem_init.h>
#include <oskar_mem_init_copy.h>
#include <oskar_vector_types.h>


TEST(Mem, add_matrix_cpu)
{
    // Use case: Two CPU oskar_Mem matrix types are added together.
    int num_elements = 10, status = 0;
    oskar_Mem mem_A, mem_B, mem_C;
    oskar_mem_init(&mem_A, OSKAR_SINGLE_COMPLEX_MATRIX,
            OSKAR_LOCATION_CPU, num_elements, 1, &status);
    oskar_mem_init(&mem_B, OSKAR_SINGLE_COMPLEX_MATRIX,
            OSKAR_LOCATION_CPU, num_elements, 1, &status);
    float4c* A = (float4c*)mem_A.data;
    float4c* B = (float4c*)mem_B.data;

    for (int i = 0; i < num_elements; ++i)
    {
        A[i].a.x = (float)i + 0.1;
        A[i].a.y = (float)i + 0.2;
        A[i].b.x = (float)i + 0.3;
        A[i].b.y = (float)i + 0.4;
        A[i].c.x = (float)i + 0.5;
        A[i].c.y = (float)i + 0.6;
        A[i].d.x = (float)i + 0.7;
        A[i].d.y = (float)i + 0.8;
        B[i].a.x = 1.15;
        B[i].a.y = 0.15;
        B[i].b.x = 2.16;
        B[i].b.y = 0.16;
        B[i].c.x = 3.17;
        B[i].c.y = 0.17;
        B[i].d.x = 4.18;
        B[i].d.y = 0.18;
    }

    oskar_mem_init(&mem_C, OSKAR_SINGLE_COMPLEX_MATRIX,
            OSKAR_LOCATION_CPU, num_elements, 1, &status);
    oskar_mem_add(&mem_C, &mem_A, &mem_B, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    float4c* C = (float4c*)mem_C.data;
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
    oskar_mem_free(&mem_A, &status);
    oskar_mem_free(&mem_B, &status);
    oskar_mem_free(&mem_C, &status);
}


TEST(Mem, add_in_place)
{
    // Use case: In place add.
    int num_elements = 10, status = 0;
    oskar_Mem mem_A, mem_B;
    oskar_mem_init(&mem_A, OSKAR_SINGLE_COMPLEX_MATRIX,
            OSKAR_LOCATION_CPU, num_elements, 1, &status);
    oskar_mem_init(&mem_B, OSKAR_SINGLE_COMPLEX_MATRIX,
            OSKAR_LOCATION_CPU, num_elements, 1, &status);
    float4c* A = (float4c*)mem_A.data;
    float4c* B = (float4c*)mem_B.data;

    for (int i = 0; i < num_elements; ++i)
    {
        A[i].a.x = (float)i + 0.1;
        A[i].a.y = (float)i + 0.2;
        A[i].b.x = (float)i + 0.3;
        A[i].b.y = (float)i + 0.4;
        A[i].c.x = (float)i + 0.5;
        A[i].c.y = (float)i + 0.6;
        A[i].d.x = (float)i + 0.7;
        A[i].d.y = (float)i + 0.8;
        EXPECT_FLOAT_EQ(0.0, B[i].a.x);
        EXPECT_FLOAT_EQ(0.0, B[i].a.y);
        EXPECT_FLOAT_EQ(0.0, B[i].b.x);
        EXPECT_FLOAT_EQ(0.0, B[i].b.y);
        EXPECT_FLOAT_EQ(0.0, B[i].c.x);
        EXPECT_FLOAT_EQ(0.0, B[i].c.y);
        EXPECT_FLOAT_EQ(0.0, B[i].d.x);
        EXPECT_FLOAT_EQ(0.0, B[i].d.y);
    }

    oskar_mem_add(&mem_B, &mem_A, &mem_B, &status);
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
    oskar_mem_free(&mem_A, &status);
    oskar_mem_free(&mem_B, &status);
}


TEST(Mem, add_gpu)
{
    // Use Case: memory on the GPU.
    int num_elements = 10, status = 0;
    oskar_Mem mem_A, mem_B, mem_C;
    oskar_mem_init(&mem_A, OSKAR_SINGLE_COMPLEX_MATRIX,
            OSKAR_LOCATION_GPU, num_elements, 1, &status);
    oskar_mem_init_copy(&mem_B, &mem_A, OSKAR_LOCATION_GPU, &status);
    oskar_mem_init_copy(&mem_C, &mem_A, OSKAR_LOCATION_GPU, &status);
    oskar_mem_add(&mem_C, &mem_A, &mem_B, &status);
    ASSERT_EQ((int)OSKAR_ERR_BAD_LOCATION, status);
    status = 0;
    oskar_mem_free(&mem_A, &status);
    oskar_mem_free(&mem_B, &status);
    oskar_mem_free(&mem_C, &status);
}


TEST(Mem, add_dimension_mismatch)
{
    // Use Case: Dimension mismatch in arrays being added.
    int num_elements = 10, status = 0;
    oskar_Mem mem_A, mem_B, mem_C;
    oskar_mem_init(&mem_A, OSKAR_SINGLE_COMPLEX_MATRIX,
            OSKAR_LOCATION_CPU, num_elements, 1, &status);
    oskar_mem_init_copy(&mem_B, &mem_A, OSKAR_LOCATION_CPU, &status);
    oskar_mem_init(&mem_C, OSKAR_SINGLE_COMPLEX_MATRIX,
            OSKAR_LOCATION_CPU, num_elements / 2, 1, &status);
    oskar_mem_add(&mem_C, &mem_A, &mem_B, &status);
    ASSERT_EQ((int)OSKAR_ERR_DIMENSION_MISMATCH, status);
    status = 0;
    oskar_mem_free(&mem_A, &status);
    oskar_mem_free(&mem_B, &status);
    oskar_mem_free(&mem_C, &status);
}
