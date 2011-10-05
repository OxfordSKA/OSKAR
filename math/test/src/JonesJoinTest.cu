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

/**
 * @details
 * Tests multiplication of Jones matrices using CUDA (single precision).
 */
void JonesJoinTest::test_float4c_inline()
{
    // Set-up some test parameters.
    int n_src = 10, n_stat = 25;
    oskar_Jones* j1 = construct_jones_device(OSKAR_JONES_FLOAT_MATRIX,
            n_src, n_stat, 0);
    oskar_Jones* j2 = construct_jones_device(OSKAR_JONES_FLOAT_MATRIX,
            n_src, n_stat, 1);

    // Call wrapper function.
    fail_on_error ( j1->join_right(j2) );

    // Check result.
    check_matrix_matrix(j1);

    // Free memory.
    delete j1;
    delete j2;
}

/**
 * @details
 * Tests multiplication of Jones matrices using CUDA (double precision).
 */
void JonesJoinTest::test_double4c_inline()
{
    if (!oskar_cuda_device_supports_double(0)) return;

    // Set-up some test parameters.
    int n_src = 10, n_stat = 25;
    oskar_Jones* j1 = construct_jones_device(OSKAR_JONES_DOUBLE_MATRIX,
            n_src, n_stat, 0);
    oskar_Jones* j2 = construct_jones_device(OSKAR_JONES_DOUBLE_MATRIX,
            n_src, n_stat, 1);

    // Call wrapper function.
    fail_on_error ( j1->join_right(j2) );

    // Check result.
    check_matrix_matrix(j1);

    // Free memory.
    delete j1;
    delete j2;
}

/*=============================================================================
 * Private members
 *---------------------------------------------------------------------------*/

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
void JonesJoinTest::construct_double4c_output_matrix_matrix(int i, int j,
        double4c& m)
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
void JonesJoinTest::construct_double4c_output_matrix_scalar(int i, int j,
        double4c& m)
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
void JonesJoinTest::construct_double4c_output_scalar_scalar(int i, int j,
        double2& m)
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
void JonesJoinTest::construct_float4c_output_matrix_matrix(int i, int j,
        float4c& m)
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
void JonesJoinTest::construct_float4c_output_matrix_scalar(int i, int j,
        float4c& m)
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
void JonesJoinTest::construct_float4c_output_scalar_scalar(int i, int j,
        float2& m)
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
oskar_Jones* JonesJoinTest::construct_jones_host(int type, int n_src,
        int n_stat, int offset)
{
    oskar_Jones* m = new oskar_Jones(type, n_src, n_stat, 0);
    int n = n_src * n_stat;
    if (type == OSKAR_JONES_FLOAT_MATRIX)
    {
        float4c* ptr = (float4c*)m->data;
        for (int i = 0; i < n; ++i)
            construct_float4c_input(i, ptr[i]);
    }
    else if (type == OSKAR_JONES_DOUBLE_MATRIX)
    {
        double4c* ptr = (double4c*)m->data;
        for (int i = 0; i < n; ++i)
            construct_double4c_input(i, ptr[i]);
    }
    else if (type == OSKAR_JONES_FLOAT_SCALAR)
    {
        float2* ptr = (float2*)m->data;
        for (int i = 0; i < n; ++i)
            construct_float2_input(i, ptr[i]);
    }
    else if (type == OSKAR_JONES_DOUBLE_SCALAR)
    {
        double2* ptr = (double2*)m->data;
        for (int i = 0; i < n; ++i)
            construct_double2_input(i, ptr[i]);
    }

    // Return a pointer to the matrix structure.
    return m;
}

/**
 * @details
 * Returns a populated Jones matrix in device memory.
 */
oskar_Jones* JonesJoinTest::construct_jones_device(int type, int n_src,
        int n_stat, int offset)
{
    // Get the matrix in host memory.
    oskar_Jones* t = construct_jones_host(type, n_src, n_stat);

    // Copy the data to device memory.
    oskar_Jones* m = new oskar_Jones(t, 1);

    // Delete the temporary and return the matrix structure.
    delete t;
    return m;
}

/**
 * @details
 * Checks result after performing matrix-matrix multiply.
 */
void JonesJoinTest::check_matrix_matrix(const oskar_Jones* jones,
        int offset1, int offset2)
{
    // Copy result to temporary host buffer.
    const oskar_Jones* temp = (jones->location() == 1) ?
            new oskar_Jones(jones, 0) : jones;

    // Check the contents of the host buffer.
    int n = jones->n_sources() * jones->n_stations();
    if (jones->type() == OSKAR_JONES_FLOAT_MATRIX)
    {
        float4c* ptr = (float4c*)temp->data;
        float4c t;
        for (int i = 0; i < n; ++i)
        {
            construct_float4c_output_matrix_matrix(i + offset1,
                    i + offset2, t);
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
        double4c* ptr = (double4c*)temp->data;
        double4c t;
        for (int i = 0; i < n; ++i)
        {
            construct_double4c_output_matrix_matrix(i + offset1,
                    i + offset2, t);
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
    if (jones->location() == 1) delete temp;
}

/**
 * @details
 * Checks result after performing matrix-scalar multiply.
 */
void JonesJoinTest::check_matrix_scalar(const oskar_Jones* data,
        int offset1, int offset2)
{

}

/**
 * @details
 * Checks result after performing scalar-scalar multiply.
 */
void JonesJoinTest::check_scalar_scalar(const oskar_Jones* data,
        int offset1, int offset2)
{

}

/**
 * @details
 * Checks for errors and fails the test if one is found.
 */
void JonesJoinTest::fail_on_error(int err)
{
    if (err > 0)
        CPPUNIT_FAIL("CUDA error, code " + oskar_to_std_string(err) +
                ": " + cudaGetErrorString((cudaError_t)err));
    else if (err < 0)
        CPPUNIT_FAIL("OSKAR error, code " + oskar_to_std_string(err));
}
