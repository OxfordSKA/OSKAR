/*
 * Copyright (c) 2011, The University of Oxford
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

#include "math/test/CudaJonesMultiplyTest.h"
#include "math/cudak/oskar_cudak_jones_mul_mat1_c2.h"
#include "math/cudak/oskar_cudak_jones_mul_mat2_c2.h"
#include "math/cudak/oskar_cudak_jones_mul_mat2.h"
#include "utility/oskar_vector_types.h"

#include <cublas.h>

#define TIMER_ENABLE 1
#include "utility/timer.h"

/**
 * @details
 * Sets up the context before running each test method.
 */
void CudaJonesMultiplyTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void CudaJonesMultiplyTest::tearDown()
{
}

/**
 * @details
 * Tests multiplication of Jones matrices using CUDA.
 */
void CudaJonesMultiplyTest::test_mat1_c2()
{
}

/**
 * @details
 * Tests multiplication of Jones matrices using CUDA.
 */
void CudaJonesMultiplyTest::test_mat2_c2()
{

}

/**
 * @details
 * Tests multiplication of Jones matrices using CUDA (single precision).
 */
void CudaJonesMultiplyTest::test_mat2_f()
{
    // Size of matrix arrays.
    int n = 5000000;
    float4c* h_m1 = (float4c*)malloc(n * sizeof(float4c));
    float4c* h_m2 = (float4c*)malloc(n * sizeof(float4c));
    float4c* h_result = (float4c*)malloc(n * sizeof(float4c));

    // Fill matrices.
    h_m1[0].a = make_float2(1.0f, 2.0f);
    h_m1[0].b = make_float2(3.0f, 4.0f);
    h_m1[0].c = make_float2(5.0f, 6.0f);
    h_m1[0].d = make_float2(7.0f, 8.0f);
    h_m2[0].a = make_float2(11.0f, 12.0f);
    h_m2[0].b = make_float2(13.0f, 14.0f);
    h_m2[0].c = make_float2(15.0f, 16.0f);
    h_m2[0].d = make_float2(17.0f, 18.0f);
    h_m1[1].a = make_float2(2.0f, 4.0f);
    h_m1[1].b = make_float2(6.0f, 8.0f);
    h_m1[1].c = make_float2(10.0f, 12.0f);
    h_m1[1].d = make_float2(14.0f, 16.0f);
    h_m2[1].a = make_float2(33.0f, 36.0f);
    h_m2[1].b = make_float2(39.0f, 42.0f);
    h_m2[1].c = make_float2(45.0f, 48.0f);
    h_m2[1].d = make_float2(51.0f, -54.0f);

    // Copy to device.
    float4c *d_m1, *d_m2, *d_result;
    cudaMalloc((void**)&d_m1, n * sizeof(float4c));
    cudaMalloc((void**)&d_m2, n * sizeof(float4c));
    cudaMalloc((void**)&d_result, n * sizeof(float4c));
    cudaMemcpy(d_m1, h_m1, n * sizeof(float4c), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, h_m2, n * sizeof(float4c), cudaMemcpyHostToDevice);

    // Invoke kernel.
    int n_thd = 256;
    int n_blk = (n + n_thd - 1) / n_thd;
    TIMER_START
    oskar_cudak_jones_mul_mat2_f <<< n_blk, n_thd >>> (n, d_m1, d_m2, d_result);
    cudaDeviceSynchronize();
    TIMER_STOP("Finished Jones matrix multiplication (single), %d matrices", n)
    int err = cudaPeekAtLastError();
    if (err)
    {
        printf("CUDA Error, code %d\n", err);
        CPPUNIT_FAIL("CUDA Error");
    }

    // Copy memory back to host.
    cudaMemcpy(h_result, d_result, n * sizeof(float4c), cudaMemcpyDeviceToHost);

    // Check contents of memory.
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-32.0, h_result[0].a.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(142.0, h_result[0].a.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-36.0, h_result[0].b.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(162.0, h_result[0].b.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-40.0, h_result[0].c.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(358.0, h_result[0].c.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-44.0, h_result[0].d.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(410.0, h_result[0].d.y, 0.001);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-192.0, h_result[1].a.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(852.0, h_result[1].a.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(648.0, h_result[1].b.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(324.0, h_result[1].b.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-240.0, h_result[1].c.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2148.0, h_result[1].c.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1464.0, h_result[1].d.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(948.0, h_result[1].d.y, 0.001);

    // Free memory.
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_result);
    free(h_m1);
    free(h_m2);
    free(h_result);
}

/**
 * @details
 * Tests multiplication of Jones matrices using CUDA (double precision).
 */
void CudaJonesMultiplyTest::test_mat2_d()
{
    // Size of matrix arrays.
    int n = 5000000;
    double4c* h_m1 = (double4c*)malloc(n * sizeof(double4c));
    double4c* h_m2 = (double4c*)malloc(n * sizeof(double4c));
    double4c* h_result = (double4c*)malloc(n * sizeof(double4c));

    // Fill matrices.
    h_m1[0].a = make_double2(1.0, 2.0);
    h_m1[0].b = make_double2(3.0, 4.0);
    h_m1[0].c = make_double2(5.0, 6.0);
    h_m1[0].d = make_double2(7.0, 8.0);
    h_m2[0].a = make_double2(11.0, 12.0);
    h_m2[0].b = make_double2(13.0, 14.0);
    h_m2[0].c = make_double2(15.0, 16.0);
    h_m2[0].d = make_double2(17.0, 18.0);
    h_m1[1].a = make_double2(2.0, 4.0);
    h_m1[1].b = make_double2(6.0, 8.0);
    h_m1[1].c = make_double2(10.0, 12.0);
    h_m1[1].d = make_double2(14.0, 16.0);
    h_m2[1].a = make_double2(33.0, 36.0);
    h_m2[1].b = make_double2(39.0, 42.0);
    h_m2[1].c = make_double2(45.0, 48.0);
    h_m2[1].d = make_double2(51.0, -54.0);

    // Copy to device.
    double4c *d_m1, *d_m2, *d_result;
    cudaMalloc((void**)&d_m1, n * sizeof(double4c));
    cudaMalloc((void**)&d_m2, n * sizeof(double4c));
    cudaMalloc((void**)&d_result, n * sizeof(double4c));
    cudaMemcpy(d_m1, h_m1, n * sizeof(double4c), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, h_m2, n * sizeof(double4c), cudaMemcpyHostToDevice);

    // Invoke kernel.
    int n_thd = 256;
    int n_blk = (n + n_thd - 1) / n_thd;
    TIMER_START
    oskar_cudak_jones_mul_mat2_d <<< n_blk, n_thd >>> (n, d_m1, d_m2, d_result);
    cudaDeviceSynchronize();
    TIMER_STOP("Finished Jones matrix multiplication (double), %d matrices", n)
    int err = cudaPeekAtLastError();
    if (err)
    {
        printf("CUDA Error, code %d\n", err);
        CPPUNIT_FAIL("CUDA Error");
    }

    // Copy memory back to host.
    cudaMemcpy(h_result, d_result, n * sizeof(double4c), cudaMemcpyDeviceToHost);

    // Check contents of memory.
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-32.0, h_result[0].a.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(142.0, h_result[0].a.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-36.0, h_result[0].b.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(162.0, h_result[0].b.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-40.0, h_result[0].c.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(358.0, h_result[0].c.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-44.0, h_result[0].d.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(410.0, h_result[0].d.y, 0.001);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-192.0, h_result[1].a.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(852.0, h_result[1].a.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(648.0, h_result[1].b.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(324.0, h_result[1].b.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-240.0, h_result[1].c.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2148.0, h_result[1].c.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1464.0, h_result[1].d.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(948.0, h_result[1].d.y, 0.001);

    // Free memory.
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_result);
    free(h_m1);
    free(h_m2);
    free(h_result);
}

/**
 * @details
 * Tests multiplication of Jones matrices using CUDA (single precision).
 */
void CudaJonesMultiplyTest::test_mat2_blas_f()
{
    // Size of matrix arrays.
    int n = 5000000;
    float4c* h_m1 = (float4c*)malloc(n * sizeof(float4c));
    float4c* h_m2 = (float4c*)malloc(n * sizeof(float4c));
    float4c* h_result = (float4c*)malloc(n * sizeof(float4c));

    // Fill matrices.
    h_m1[0].a = make_float2(1.0f, 2.0f);
    h_m1[0].c = make_float2(3.0f, 4.0f);
    h_m1[0].b = make_float2(5.0f, 6.0f);
    h_m1[0].d = make_float2(7.0f, 8.0f);
    h_m2[0].a = make_float2(11.0f, 12.0f);
    h_m2[0].c = make_float2(13.0f, 14.0f);
    h_m2[0].b = make_float2(15.0f, 16.0f);
    h_m2[0].d = make_float2(17.0f, 18.0f);
    h_m1[1].a = make_float2(2.0f, 4.0f);
    h_m1[1].c = make_float2(6.0f, 8.0f);
    h_m1[1].b = make_float2(10.0f, 12.0f);
    h_m1[1].d = make_float2(14.0f, 16.0f);
    h_m2[1].a = make_float2(33.0f, 36.0f);
    h_m2[1].c = make_float2(39.0f, 42.0f);
    h_m2[1].b = make_float2(45.0f, 48.0f);
    h_m2[1].d = make_float2(51.0f, -54.0f);

    // Copy to device.
    float4c *d_m1, *d_m2, *d_result;
    cudaMalloc((void**)&d_m1, n * sizeof(float4c));
    cudaMalloc((void**)&d_m2, n * sizeof(float4c));
    cudaMalloc((void**)&d_result, n * sizeof(float4c));
    cudaMemcpy(d_m1, h_m1, n * sizeof(float4c), cudaMemcpyHostToDevice);
    cudaMemcpy(d_m2, h_m2, n * sizeof(float4c), cudaMemcpyHostToDevice);

    // Invoke kernel.
    cublasInit();
    cuComplex one = make_cuComplex(1.0f, 0.0f);
    cuComplex zero = make_cuComplex(0.0f, 0.0f);
    TIMER_START
    for (int i = 0; i < n; ++i)
    {
        cublasCgemm('n', 'n', 2, 2, 2, one, (const cuComplex*)&d_m1[i], 2,
                (const cuComplex*)&d_m2[i], 2,
                zero, (cuComplex*)&d_result[i], 2);
    }
    TIMER_STOP("Finished Jones matrix multiplication (single, CUBLAS), %d matrices", n)
    cublasShutdown();
    int err = cudaPeekAtLastError();
    if (err)
    {
        printf("CUDA Error, code %d\n", err);
        CPPUNIT_FAIL("CUDA Error");
    }

    // Copy memory back to host.
    cudaMemcpy(h_result, d_result, n * sizeof(float4c), cudaMemcpyDeviceToHost);

    // Check contents of memory.
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-32.0, h_result[0].a.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(142.0, h_result[0].a.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-36.0, h_result[0].c.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(162.0, h_result[0].c.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-40.0, h_result[0].b.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(358.0, h_result[0].b.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-44.0, h_result[0].d.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(410.0, h_result[0].d.y, 0.001);

    CPPUNIT_ASSERT_DOUBLES_EQUAL(-192.0, h_result[1].a.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(852.0, h_result[1].a.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(648.0, h_result[1].c.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(324.0, h_result[1].c.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(-240.0, h_result[1].b.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2148.0, h_result[1].b.y, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(1464.0, h_result[1].d.x, 0.001);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(948.0, h_result[1].d.y, 0.001);

    // Free memory.
    cudaFree(d_m1);
    cudaFree(d_m2);
    cudaFree(d_result);
    free(h_m1);
    free(h_m2);
    free(h_result);
}

