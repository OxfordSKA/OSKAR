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

#include "math/test/CudaJonesJoinTest.h"
#include "math/oskar_jones_join.h"
#include "utility/oskar_vector_types.h"
#include "utility/oskar_cuda_device_info.h"

#define TIMER_ENABLE 1
#include "utility/timer.h"

/**
 * @details
 * Returns a populated Jones matrix in host memory.
 */
oskar_Jones JonesJoinTest::jonesHost(int type, int n_src, int n_stat)
{
	oskar_Jones m(type, n_src, n_stat, 0);
	int n = n_src * n_stat;
	if (type == OSKAR_JONES_FLOAT_MATRIX)
	{
		float4c* ptr = (float4c*)malloc(n * sizeof(float4c));
		m.data = (void*)ptr;
		for (int i = 0; i < n; ++i)
		{
			int b = 8*i;
			ptr[i].a = make_float2(b + 0, b + 1);
			ptr[i].b = make_float2(b + 2, b + 3);
			ptr[i].c = make_float2(b + 4, b + 5);
			ptr[i].d = make_float2(b + 6, b + 7);
		}
	}
	else if (type == OSKAR_JONES_DOUBLE_MATRIX)
	{
		double4c* ptr = (double4c*)malloc(n * sizeof(double4c));
		m.data = (void*)ptr;
		for (int i = 0; i < n; ++i)
		{
			int b = 8*i;
			ptr[i].a = make_double2(b + 0, b + 1);
			ptr[i].b = make_double2(b + 2, b + 3);
			ptr[i].c = make_double2(b + 4, b + 5);
			ptr[i].d = make_double2(b + 6, b + 7);
		}
	}
	else if (type == OSKAR_JONES_FLOAT_SCALAR)
	{
		float2* ptr = (float2*)malloc(n * sizeof(float2));
		m.data = (void*)ptr;
		for (int i = 0; i < n; ++i)
			ptr[i] = make_float2(i, i);
	}
	else if (type == OSKAR_JONES_DOUBLE_SCALAR)
	{
		double2* ptr = (double2*)malloc(n * sizeof(double2));
		m.data = (void*)ptr;
		for (int i = 0; i < n; ++i)
			ptr[i] = make_double2(i, i);
	}

	// Return the matrix structure.
	return m;
}

/**
 * @details
 * Returns a populated Jones matrix in device memory.
 */
oskar_Jones JonesJoinTest::jonesDevice(int type, int n_src, int n_stat)
{
	// Get the matrix in host memory.
	oskar_Jones h_m = jonesHost(type, n_src, n_stat);

	// Copy host to device memory.
	oskar_Jones d_m(type, n_src, n_stat, 1);
	int n = n_src * n_stat;
	if (type == OSKAR_JONES_FLOAT_MATRIX)
	{
		cudaMalloc(&d_m.data, n * sizeof(float4c));
		cudaMemcpy(d_m.data, h_m.data, n * sizeof(float4c),
				cudaMemcpyHostToDevice);
	}
	else if (type == OSKAR_JONES_DOUBLE_MATRIX)
	{
		cudaMalloc(&d_m.data, n * sizeof(double4c));
		cudaMemcpy(d_m.data, h_m.data, n * sizeof(double4c),
				cudaMemcpyHostToDevice);
	}
	else if (type == OSKAR_JONES_FLOAT_SCALAR)
	{
		cudaMalloc(&d_m.data, n * sizeof(float2));
		cudaMemcpy(d_m.data, h_m.data, n * sizeof(float2),
				cudaMemcpyHostToDevice);
	}
	else if (type == OSKAR_JONES_DOUBLE_SCALAR)
	{
		cudaMalloc(&d_m.data, n * sizeof(double2));
		cudaMemcpy(d_m.data, h_m.data, n * sizeof(double2),
				cudaMemcpyHostToDevice);
	}

	// Free host memory.
	free(h_m.data);

	// Return the matrix structure.
	return d_m;
}

/**
 * @details
 * Checks result after performing matrix-matrix multiply.
 */
void JonesJoinTest::checkResultMatrixMatrix(const oskar_Jones* jones)
{
	void* h_data = NULL;
	int n = jones->n_sources() * jones->n_stations();

	// Copy result to temporary host buffer.
	if (jones->location() == 1)
	{
		int bytes = 0;
		if (jones->type() == OSKAR_JONES_FLOAT_MATRIX)
			bytes = n * sizeof(float4c);
		else if (jones->type() == OSKAR_JONES_DOUBLE_MATRIX)
			bytes = n * sizeof(double4c);
		h_data = malloc(bytes);
		cudaMemcpy(h_data, jones->data, bytes, cudaMemcpyDeviceToHost);
	}
	else h_data = jones->data;

	// Check the contents of the host buffer.
	if (jones->type() == OSKAR_JONES_FLOAT_MATRIX)
	{
		float4c* ptr = (float4c*)h_data;
		for (int i = 0; i < n; ++i)
		{
			int s = (i+1) * (i+1);
//			CPPUNIT_ASSERT_DOUBLES_EQUAL( -8 * s, ptr[i].a.x, 0.001);
//			CPPUNIT_ASSERT_DOUBLES_EQUAL( 22 * s, ptr[i].a.y, 0.001);
//			CPPUNIT_ASSERT_DOUBLES_EQUAL(-12 * s, ptr[i].b.x, 0.001);
//			CPPUNIT_ASSERT_DOUBLES_EQUAL( 34 * s, ptr[i].b.y, 0.001);
//			CPPUNIT_ASSERT_DOUBLES_EQUAL(-16 * s, ptr[i].c.x, 0.001);
//			CPPUNIT_ASSERT_DOUBLES_EQUAL( 62 * s, ptr[i].c.y, 0.001);
//			CPPUNIT_ASSERT_DOUBLES_EQUAL(-20 * s, ptr[i].d.x, 0.001);
//			CPPUNIT_ASSERT_DOUBLES_EQUAL(106 * s, ptr[i].d.y, 0.001);
		}
	}
	else if (jones->type() == OSKAR_JONES_DOUBLE_MATRIX)
	{
		double4c* ptr = (double4c*)h_data;
		for (int i = 0; i < n; ++i)
		{
			int s = (i+1) * (i+1);
//			CPPUNIT_ASSERT_DOUBLES_EQUAL( -8 * s, ptr[i].a.x, 0.001);
//			CPPUNIT_ASSERT_DOUBLES_EQUAL( 22 * s, ptr[i].a.y, 0.001);
//			CPPUNIT_ASSERT_DOUBLES_EQUAL(-12 * s, ptr[i].b.x, 0.001);
//			CPPUNIT_ASSERT_DOUBLES_EQUAL( 34 * s, ptr[i].b.y, 0.001);
//			CPPUNIT_ASSERT_DOUBLES_EQUAL(-16 * s, ptr[i].c.x, 0.001);
//			CPPUNIT_ASSERT_DOUBLES_EQUAL( 62 * s, ptr[i].c.y, 0.001);
//			CPPUNIT_ASSERT_DOUBLES_EQUAL(-20 * s, ptr[i].d.x, 0.001);
//			CPPUNIT_ASSERT_DOUBLES_EQUAL(106 * s, ptr[i].d.y, 0.001);
		}
	}

	// Free temporary host buffer.
	if (jones->location() == 1)
		free(h_data);
}

/**
 * @details
 * Checks result after performing matrix-scalar multiply.
 */
void JonesJoinTest::checkResultMatrixScalar(const oskar_Jones* data)
{

}

/**
 * @details
 * Checks result after performing scalar-scalar multiply.
 */
void JonesJoinTest::checkResultScalarScalar(const oskar_Jones* data)
{

}

/**
 * @details
 * Sets up the context before running each test method.
 */
void JonesJoinTest::setUp()
{
}

/**
 * @details
 * Clean up routine called after each test is run.
 */
void JonesJoinTest::tearDown()
{
}

/**
 * @details
 * Tests multiplication of Jones matrices using CUDA (single precision).
 */
void JonesJoinTest::test_mat2_f()
{
	int n_src = 10;
	int n_stat = 25;
	oskar_Jones j1 = jonesDevice(OSKAR_JONES_FLOAT_MATRIX, n_src, n_stat);
	oskar_Jones j2 = jonesDevice(OSKAR_JONES_FLOAT_MATRIX, n_src, n_stat);

	// Call wrapper function.
	int err = 0, n = n_src * n_stat;
	TIMER_START
	err = oskar_jones_join(NULL, &j1, &j2); // J1 = J1 * J2
	TIMER_STOP("Finished Jones matrix join (float), %d matrices", n)
	if (err)
	{
		printf("CUDA Error, code %d\n", err);
		CPPUNIT_FAIL("CUDA Error");
	}

	// Check result.
	checkResultMatrixMatrix(&j1);

	// Free memory.
	cudaFree(j1.data);
	cudaFree(j2.data);

//    // Size of matrix arrays.
//    int n = 500000;
//    float4c* h_m1 = (float4c*)malloc(n * sizeof(float4c));
//    float4c* h_m2 = (float4c*)malloc(n * sizeof(float4c));
//    float4c* h_result = (float4c*)malloc(n * sizeof(float4c));
//
//    // Fill matrices.
//    h_m1[0].a = make_float2(1.0f, 2.0f);
//    h_m1[0].b = make_float2(3.0f, 4.0f);
//    h_m1[0].c = make_float2(5.0f, 6.0f);
//    h_m1[0].d = make_float2(7.0f, 8.0f);
//    h_m2[0].a = make_float2(11.0f, 12.0f);
//    h_m2[0].b = make_float2(13.0f, 14.0f);
//    h_m2[0].c = make_float2(15.0f, 16.0f);
//    h_m2[0].d = make_float2(17.0f, 18.0f);
//    h_m1[1].a = make_float2(2.0f, 4.0f);
//    h_m1[1].b = make_float2(6.0f, 8.0f);
//    h_m1[1].c = make_float2(10.0f, 12.0f);
//    h_m1[1].d = make_float2(14.0f, 16.0f);
//    h_m2[1].a = make_float2(33.0f, 36.0f);
//    h_m2[1].b = make_float2(39.0f, 42.0f);
//    h_m2[1].c = make_float2(45.0f, 48.0f);
//    h_m2[1].d = make_float2(51.0f, -54.0f);
//
//    // Copy to device.
//    float4c *d_m1, *d_m2;
//    cudaMalloc((void**)&d_m1, n * sizeof(float4c));
//    cudaMalloc((void**)&d_m2, n * sizeof(float4c));
//    cudaMemcpy(d_m1, h_m1, n * sizeof(float4c), cudaMemcpyHostToDevice);
//    cudaMemcpy(d_m2, h_m2, n * sizeof(float4c), cudaMemcpyHostToDevice);
//
//    // Set up containers.
//    oskar_Jones j1(OSKAR_JONES_FLOAT_MATRIX, 50000, 10, 1);
//    j1.data = (void*)d_m1;
//    oskar_Jones j2(OSKAR_JONES_FLOAT_MATRIX, 50000, 10, 1);
//    j2.data = (void*)d_m2;
//
//    // Call wrapper function.
//    int err = 0;
//    TIMER_START
//    err = oskar_jones_join(NULL, &j1, &j2); // J1 = J1 * J2
//    TIMER_STOP("Finished Jones matrix join (float), %d matrices", n)
//    if (err)
//    {
//        printf("CUDA Error, code %d\n", err);
//        CPPUNIT_FAIL("CUDA Error");
//    }
//
//    // Copy memory back to host.
//    cudaMemcpy(h_result, d_m1, n * sizeof(float4c), cudaMemcpyDeviceToHost);
//
//    // Check contents of memory.
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(-32.0, h_result[0].a.x, 0.001);
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(142.0, h_result[0].a.y, 0.001);
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(-36.0, h_result[0].b.x, 0.001);
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(162.0, h_result[0].b.y, 0.001);
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(-40.0, h_result[0].c.x, 0.001);
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(358.0, h_result[0].c.y, 0.001);
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(-44.0, h_result[0].d.x, 0.001);
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(410.0, h_result[0].d.y, 0.001);
//
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(-192.0, h_result[1].a.x, 0.001);
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(852.0, h_result[1].a.y, 0.001);
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(648.0, h_result[1].b.x, 0.001);
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(324.0, h_result[1].b.y, 0.001);
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(-240.0, h_result[1].c.x, 0.001);
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(2148.0, h_result[1].c.y, 0.001);
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(1464.0, h_result[1].d.x, 0.001);
//    CPPUNIT_ASSERT_DOUBLES_EQUAL(948.0, h_result[1].d.y, 0.001);
//
//    // Free memory.
//    cudaFree(d_m1);
//    cudaFree(d_m2);
//    free(h_m1);
//    free(h_m2);
//    free(h_result);
}

/**
 * @details
 * Tests multiplication of Jones matrices using CUDA (double precision).
 */
void JonesJoinTest::test_mat2_d()
{
    if (!oskar_cuda_device_supports_double(0))
        return;

    // Size of matrix arrays.
    int n = 500000;
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

    // Set up containers.
    oskar_Jones j1(OSKAR_JONES_DOUBLE_MATRIX, 50000, 10, 1);
    j1.data = (void*)d_m1;
    oskar_Jones j2(OSKAR_JONES_DOUBLE_MATRIX, 50000, 10, 1);
    j2.data = (void*)d_m2;

    // Call wrapper function.
    int err = 0;
    TIMER_START
    err = oskar_jones_join(NULL, &j1, &j2); // J1 = J1 * J2
    TIMER_STOP("Finished Jones matrix join (double), %d matrices", n)
    if (err)
    {
        printf("CUDA Error, code %d\n", err);
        CPPUNIT_FAIL("CUDA Error");
    }

    // Copy memory back to host.
    cudaMemcpy(h_result, d_m1, n * sizeof(double4c), cudaMemcpyDeviceToHost);

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
