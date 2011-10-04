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

#include "math/test/JonesJoinTest.h"
#include "math/oskar_jones_join.h"
#include "utility/oskar_cuda_device_info.h"

#define TIMER_ENABLE 1
#include "utility/timer.h"

#include <sstream>

/**
 * @details
 * Converts the parameter to a C++ string.
 */
template <class T>
inline std::string oskar_to_std_string(const T& t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}

/**
 * @details
 * Returns the sum of two complex numbers.
 */
template<typename T> T complex_add(T a, T b)
{
    // Add complex numbers a and b.
    T out;
    out.x = a.x + b.x; // RE+RE
    out.y = a.y + b.y; // IM+IM
    return out;
}

/**
 * @details
 * Returns the product of two complex numbers.
 */
template<typename T> T complex_mul(T a, T b)
{
    // Multiply complex numbers a and b.
    T out;
    out.x = a.x * b.x - a.y * b.y; // RE*RE - IM*IM
    out.y = a.x * b.y + a.y * b.x; // RE*IM + IM*RE
    return out;
}

/**
 * @details
 * Returns a single, populated Jones scalar.
 */
void JonesJoinTest::construct_double2_input(int i, double2& m)
{
    m = make_double2(i, i);
}

/**
 * @details
 * Returns a single, populated 2x2 Jones matrix.
 */
void JonesJoinTest::construct_double4c_input(int i, double4c& m)
{
    i *= 2;
    m.a = make_double2(i + 0.0, i + 0.2);
    m.b = make_double2(i + 0.4, i + 0.6);
    m.c = make_double2(i + 0.8, i + 1.0);
    m.d = make_double2(i + 1.2, i + 1.4);
}

/**
 * @details
 * Returns a single, populated 2x2 Jones matrix.
 */
void JonesJoinTest::construct_double4c_output_matrix_matrix(int i, double4c& m)
{
    // Get the input matrix.
    double4c p;
    construct_double4c_input(i, p);

    // Compute p*p.
    m.a = complex_add(complex_mul(p.a, p.a), complex_mul(p.b, p.c));
    m.b = complex_add(complex_mul(p.a, p.b), complex_mul(p.b, p.d));
    m.c = complex_add(complex_mul(p.c, p.a), complex_mul(p.d, p.c));
    m.d = complex_add(complex_mul(p.c, p.b), complex_mul(p.d, p.d));
}

/**
 * @details
 * Returns a single, populated 2x2 Jones matrix.
 */
void JonesJoinTest::construct_double4c_output_matrix_scalar(int i, double4c& m)
{
    // Get the input matrix.
    double4c p;
    construct_double4c_input(i, p);

    // Get the input scalar.
    double2 s;
    construct_double2_input(i, s);

    // Compute p*s.
    m.a = complex_mul(p.a, s);
    m.b = complex_mul(p.b, s);
    m.c = complex_mul(p.c, s);
    m.d = complex_mul(p.d, s);
}

/**
 * @details
 * Returns a single, populated 2x2 Jones scalar.
 */
void JonesJoinTest::construct_double4c_output_scalar_scalar(int i, double2& m)
{
    // Get the input scalars.
    double2 s1, s2;
    construct_double2_input(i, s1);
    construct_double2_input(i, s2);

    // Compute s1*s2.
    m = complex_mul(s1, s2);
}

/**
 * @details
 * Returns a single, populated Jones scalar.
 */
void JonesJoinTest::construct_float2_input(int i, float2& m)
{
    m = make_float2(i, i);
}

/**
 * @details
 * Returns a single, populated 2x2 Jones matrix.
 */
void JonesJoinTest::construct_float4c_input(int i, float4c& m)
{
    i *= 2;
    m.a = make_float2(i + 0.0, i + 0.2);
    m.b = make_float2(i + 0.4, i + 0.6);
    m.c = make_float2(i + 0.8, i + 1.0);
    m.d = make_float2(i + 1.2, i + 1.4);
}

/**
 * @details
 * Returns a single, populated 2x2 Jones matrix.
 */
void JonesJoinTest::construct_float4c_output_matrix_matrix(int i, float4c& m)
{
    // Get the input matrix.
    float4c p;
    construct_float4c_input(i, p);

    // Compute p*p.
    m.a = complex_add(complex_mul(p.a, p.a), complex_mul(p.b, p.c));
    m.b = complex_add(complex_mul(p.a, p.b), complex_mul(p.b, p.d));
    m.c = complex_add(complex_mul(p.c, p.a), complex_mul(p.d, p.c));
    m.d = complex_add(complex_mul(p.c, p.b), complex_mul(p.d, p.d));
}

/**
 * @details
 * Returns a single, populated 2x2 Jones matrix.
 */
void JonesJoinTest::construct_float4c_output_matrix_scalar(int i, float4c& m)
{
    // Get the input matrix.
    float4c p;
    construct_float4c_input(i, p);

    // Get the input scalar.
    float2 s;
    construct_float2_input(i, s);

    // Compute p*s.
    m.a = complex_mul(p.a, s);
    m.b = complex_mul(p.b, s);
    m.c = complex_mul(p.c, s);
    m.d = complex_mul(p.d, s);
}

/**
 * @details
 * Returns a single, populated 2x2 Jones scalar.
 */
void JonesJoinTest::construct_float4c_output_scalar_scalar(int i, float2& m)
{
    // Get the input scalars.
    float2 s1, s2;
    construct_float2_input(i, s1);
    construct_float2_input(i, s2);

    // Compute s1*s2.
    m = complex_mul(s1, s2);
}

/**
 * @details
 * Returns a populated Jones matrix in host memory.
 */
oskar_Jones JonesJoinTest::construct_jones_host(int type, int n_src,
        int n_stat)
{
    oskar_Jones m(type, n_src, n_stat, 0);
    int n = n_src * n_stat;
    if (type == OSKAR_JONES_FLOAT_MATRIX)
    {
        float4c* ptr = (float4c*)malloc(n * sizeof(float4c));
        m.data = (void*)ptr;
        for (int i = 0; i < n; ++i)
            construct_float4c_input(i, ptr[i]);
    }
    else if (type == OSKAR_JONES_DOUBLE_MATRIX)
    {
        double4c* ptr = (double4c*)malloc(n * sizeof(double4c));
        m.data = (void*)ptr;
        for (int i = 0; i < n; ++i)
            construct_double4c_input(i, ptr[i]);
    }
    else if (type == OSKAR_JONES_FLOAT_SCALAR)
    {
        float2* ptr = (float2*)malloc(n * sizeof(float2));
        m.data = (void*)ptr;
        for (int i = 0; i < n; ++i)
            construct_float2_input(i, ptr[i]);
    }
    else if (type == OSKAR_JONES_DOUBLE_SCALAR)
    {
        double2* ptr = (double2*)malloc(n * sizeof(double2));
        m.data = (void*)ptr;
        for (int i = 0; i < n; ++i)
            construct_double2_input(i, ptr[i]);
    }

    // Return the matrix structure.
    return m;
}

/**
 * @details
 * Returns a populated Jones matrix in device memory.
 */
oskar_Jones JonesJoinTest::construct_jones_device(int type, int n_src,
        int n_stat)
{
    // Get the matrix in host memory.
    oskar_Jones h_m = construct_jones_host(type, n_src, n_stat);

    // Copy host to device memory.
    oskar_Jones d_m(type, n_src, n_stat, 1);
    int bytes = n_src * n_stat;
    if (type == OSKAR_JONES_FLOAT_MATRIX)
        bytes *= sizeof(float4c);
    else if (type == OSKAR_JONES_DOUBLE_MATRIX)
        bytes *= sizeof(double4c);
    else if (type == OSKAR_JONES_FLOAT_SCALAR)
        bytes *= sizeof(float2);
    else if (type == OSKAR_JONES_DOUBLE_SCALAR)
        bytes *= sizeof(double2);
    cudaMalloc(&d_m.data, bytes);
    cudaMemcpy(d_m.data, h_m.data, bytes, cudaMemcpyHostToDevice);

    // Free host memory.
    free(h_m.data);

    // Return the matrix structure.
    return d_m;
}

/**
 * @details
 * Checks result after performing matrix-matrix multiply.
 */
void JonesJoinTest::check_matrix_matrix(const oskar_Jones* jones)
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
        float4c t;
        for (int i = 0; i < n; ++i)
        {
            construct_float4c_output_matrix_matrix(i, t);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(t.a.x, ptr[i].a.x, abs(t.a.x * 4e-5));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(t.a.y, ptr[i].a.y, abs(t.a.y * 4e-5));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(t.b.x, ptr[i].b.x, abs(t.b.x * 4e-5));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(t.b.y, ptr[i].b.y, abs(t.b.y * 4e-5));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(t.c.x, ptr[i].c.x, abs(t.c.x * 4e-5));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(t.c.y, ptr[i].c.y, abs(t.c.y * 4e-5));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(t.d.x, ptr[i].d.x, abs(t.d.x * 4e-5));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(t.d.y, ptr[i].d.y, abs(t.d.y * 4e-5));
        }
    }
    else if (jones->type() == OSKAR_JONES_DOUBLE_MATRIX)
    {
        double4c* ptr = (double4c*)h_data;
        double4c t;
        for (int i = 0; i < n; ++i)
        {
            construct_double4c_output_matrix_matrix(i, t);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(t.a.x, ptr[i].a.x, abs(t.a.x * 1e-10));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(t.a.y, ptr[i].a.y, abs(t.a.y * 1e-10));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(t.b.x, ptr[i].b.x, abs(t.b.x * 1e-10));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(t.b.y, ptr[i].b.y, abs(t.b.y * 1e-10));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(t.c.x, ptr[i].c.x, abs(t.c.x * 1e-10));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(t.c.y, ptr[i].c.y, abs(t.c.y * 1e-10));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(t.d.x, ptr[i].d.x, abs(t.d.x * 1e-10));
            CPPUNIT_ASSERT_DOUBLES_EQUAL(t.d.y, ptr[i].d.y, abs(t.d.y * 1e-10));
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
void JonesJoinTest::check_matrix_scalar(const oskar_Jones* data)
{

}

/**
 * @details
 * Checks result after performing scalar-scalar multiply.
 */
void JonesJoinTest::check_scalar_scalar(const oskar_Jones* data)
{

}

/**
 * @details
 * Tests multiplication of Jones matrices using CUDA (single precision).
 */
void JonesJoinTest::test_float4c()
{
    // Set-up some test parameters.
    int n_src = 10, n_stat = 25;
    oskar_Jones j1 = construct_jones_device(OSKAR_JONES_FLOAT_MATRIX,
            n_src, n_stat);
    oskar_Jones j2 = construct_jones_device(OSKAR_JONES_FLOAT_MATRIX,
            n_src, n_stat);

    // Call wrapper function.
    int err = oskar_jones_join(NULL, &j1, &j2); // J1 = J1 * J2
    if (err > 0)
        CPPUNIT_FAIL(std::string("CUDA error, code ") +
                oskar_to_std_string(err) +
                ": " + cudaGetErrorString((cudaError_t)err));
    else if (err < 0)
        CPPUNIT_FAIL(std::string("OSKAR error, code ") +
                oskar_to_std_string(err));

    // Check result.
    check_matrix_matrix(&j1);

    // Free memory.
    cudaFree(j1.data);
    cudaFree(j2.data);
}

/**
 * @details
 * Tests multiplication of Jones matrices using CUDA (double precision).
 */
void JonesJoinTest::test_double4c()
{
    if (!oskar_cuda_device_supports_double(0))
        return;

    // Set-up some test parameters.
    int n_src = 10, n_stat = 25;
    oskar_Jones j1 = construct_jones_device(OSKAR_JONES_DOUBLE_MATRIX,
            n_src, n_stat);
    oskar_Jones j2 = construct_jones_device(OSKAR_JONES_DOUBLE_MATRIX,
            n_src, n_stat);

    // Call wrapper function.
    int err = oskar_jones_join(NULL, &j1, &j2); // J1 = J1 * J2
    if (err > 0)
        CPPUNIT_FAIL(std::string("CUDA error, code ") +
                oskar_to_std_string(err) +
                ": " + cudaGetErrorString((cudaError_t)err));
    else if (err < 0)
        CPPUNIT_FAIL(std::string("OSKAR error, code ") +
                oskar_to_std_string(err));

    // Check result.
    check_matrix_matrix(&j1);

    // Free memory.
    cudaFree(j1.data);
    cudaFree(j2.data);
}
