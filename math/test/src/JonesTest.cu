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

#include "math/test/JonesTest.h"
#include "utility/oskar_cuda_device_info.h"

#define TIMER_ENABLE 1
#include "utility/timer.h"

#include <cstdio>

/**
 * @details
 * Tests inline multiplication of Jones matrices using CUDA.
 * Both matrices are on the device.
 */
void JonesTest::test_join_inline_mat_mat_device()
{
//    return;
    // Set-up some test parameters.
    int n_src = 10, n_stat = 25;

    // Single precision test.
    {
        oskar_Jones* j1 = construct_jones_host(OSKAR_JONES_FLOAT_MATRIX,
                n_src, n_stat, 0);
        oskar_Jones* j2 = construct_jones_device(OSKAR_JONES_FLOAT_MATRIX,
                n_src, n_stat, 1);

        // Call wrapper function.
        fail_on_error ( j1->join_from_right(j2) );

        // Check result.
        check_matrix_matrix(j1, 0, 1);

        // Free memory.
        delete j1;
        delete j2;
    }

    // Return if no double precision support.
    if (!oskar_cuda_device_supports_double(0)) return;

    // Double precision test.
    {
        oskar_Jones* j1 = construct_jones_host(OSKAR_JONES_DOUBLE_MATRIX,
                n_src, n_stat, 0);
        oskar_Jones* j2 = construct_jones_device(OSKAR_JONES_DOUBLE_MATRIX,
                n_src, n_stat, 1);

        // Call wrapper function.
        fail_on_error ( j1->join_from_right(j2) );

        // Check result.
        check_matrix_matrix(j1, 0, 1);

        // Free memory.
        delete j1;
        delete j2;
    }
}

/**
 * @details
 * Tests inline multiplication of Jones matrices using CUDA.
 * One set of matrices is on the device, and the other on the host.
 */
void JonesTest::test_join_inline_mat_mat_device_host()
{
    // Set-up some test parameters.
    int n_src = 10, n_stat = 25;

    // Single precision test.
    {
        oskar_Jones* j1 = construct_jones_device(OSKAR_JONES_FLOAT_MATRIX,
                n_src, n_stat, 0);
        oskar_Jones* j2 = construct_jones_host(OSKAR_JONES_FLOAT_MATRIX,
                n_src, n_stat, 1);

        // Call wrapper function.
        fail_on_error ( j1->join_from_right(j2) );

        // Check result.
        check_matrix_matrix(j1, 0, 1);

        // Free memory.
        delete j1;
        delete j2;
    }

    // Return if no double precision support.
    if (!oskar_cuda_device_supports_double(0)) return;

    // Double precision test.
    {
        oskar_Jones* j1 = construct_jones_device(OSKAR_JONES_DOUBLE_MATRIX,
                n_src, n_stat, 0);
        oskar_Jones* j2 = construct_jones_host(OSKAR_JONES_DOUBLE_MATRIX,
                n_src, n_stat, 1);

        // Call wrapper function.
        fail_on_error ( j1->join_from_right(j2) );

        // Check result.
        check_matrix_matrix(j1, 0, 1);

        // Free memory.
        delete j1;
        delete j2;
    }
}

/**
 * @details
 * Tests inline multiplication of Jones matrices using CUDA.
 * Both matrices are on the host.
 */
void JonesTest::test_join_inline_mat_mat_host()
{
    // Set-up some test parameters.
    int n_src = 10, n_stat = 25;

    // Single precision test.
    {
        oskar_Jones* j1 = construct_jones_host(OSKAR_JONES_FLOAT_MATRIX,
                n_src, n_stat, 0);
        oskar_Jones* j2 = construct_jones_host(OSKAR_JONES_FLOAT_MATRIX,
                n_src, n_stat, 1);

        // Call wrapper function.
        fail_on_error ( j1->join_from_right(j2) );

        // Check result.
        check_matrix_matrix(j1, 0, 1);

        // Free memory.
        delete j1;
        delete j2;
    }

    // Return if no double precision support.
    if (!oskar_cuda_device_supports_double(0)) return;

    // Double precision test.
    {
        oskar_Jones* j1 = construct_jones_host(OSKAR_JONES_DOUBLE_MATRIX,
                n_src, n_stat, 0);
        oskar_Jones* j2 = construct_jones_host(OSKAR_JONES_DOUBLE_MATRIX,
                n_src, n_stat, 1);

        // Call wrapper function.
        fail_on_error ( j1->join_from_right(j2) );

        // Check result.
        check_matrix_matrix(j1, 0, 1);

        // Free memory.
        delete j1;
        delete j2;
    }
}

/**
 * @details
 * Tests setting the contents of the Jones matrices.
 * The data are on the device.
 */
void JonesTest::test_set_ones_device()
{
    // Set-up some test parameters.
    int n_src = 100, n_stat = 25;
    int n = n_src * n_stat;

    // Single precision test.
    {
        oskar_Jones* j1 = construct_jones_device(OSKAR_JONES_FLOAT_MATRIX,
                n_src, n_stat, 0);

        // Call wrapper function.
        fail_on_error ( j1->set_real_scalar(1.0) );

        // Copy result to CPU.
        oskar_Jones* t = new oskar_Jones(j1, 0);

        // Check result.
        float4c* ptr = (float4c*)t->data;
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0f, ptr[i].a.x, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, ptr[i].a.y, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, ptr[i].b.x, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, ptr[i].b.y, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, ptr[i].c.x, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, ptr[i].c.y, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0f, ptr[i].d.x, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, ptr[i].d.y, 1e-5);
        }

        // Free memory.
        delete t;
        delete j1;
    }

    // Return if no double precision support.
    if (!oskar_cuda_device_supports_double(0)) return;

    // Double precision test.
    {
        oskar_Jones* j1 = construct_jones_device(OSKAR_JONES_DOUBLE_MATRIX,
                n_src, n_stat, 0);

        // Call wrapper function.
        fail_on_error ( j1->set_real_scalar(1.0) );

        // Copy result to CPU.
        oskar_Jones* t = new oskar_Jones(j1, 0);

        // Check result.
        double4c* ptr = (double4c*)t->data;
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, ptr[i].a.x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ptr[i].a.y, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ptr[i].b.x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ptr[i].b.y, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ptr[i].c.x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ptr[i].c.y, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, ptr[i].d.x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ptr[i].d.y, 1e-12);
        }

        // Free memory.
        delete t;
        delete j1;
    }
}

/**
 * @details
 * Tests setting the contents of the Jones matrices.
 * The data are on the host.
 */
void JonesTest::test_set_ones_host()
{
    // Set-up some test parameters.
    int n_src = 10, n_stat = 25;
    int n = n_src * n_stat;

    // Single precision test.
    {
        oskar_Jones* j1 = construct_jones_host(OSKAR_JONES_FLOAT_MATRIX,
                n_src, n_stat, 0);

        // Call wrapper function.
        fail_on_error ( j1->set_real_scalar(1.0) );

        // Check result.
        float4c* ptr = (float4c*)j1->data;
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0f, ptr[i].a.x, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, ptr[i].a.y, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, ptr[i].b.x, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, ptr[i].b.y, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, ptr[i].c.x, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, ptr[i].c.y, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0f, ptr[i].d.x, 1e-5);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0f, ptr[i].d.y, 1e-5);
        }

        // Free memory.
        delete j1;
    }

    // Return if no double precision support.
    if (!oskar_cuda_device_supports_double(0)) return;

    // Double precision test.
    {
        oskar_Jones* j1 = construct_jones_host(OSKAR_JONES_DOUBLE_MATRIX,
                n_src, n_stat, 0);

        // Call wrapper function.
        fail_on_error ( j1->set_real_scalar(1.0) );

        // Check result.
        double4c* ptr = (double4c*)j1->data;
        for (int i = 0; i < n; ++i)
        {
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, ptr[i].a.x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ptr[i].a.y, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ptr[i].b.x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ptr[i].b.y, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ptr[i].c.x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ptr[i].c.y, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(1.0, ptr[i].d.x, 1e-12);
            CPPUNIT_ASSERT_DOUBLES_EQUAL(0.0, ptr[i].d.y, 1e-12);
        }

        // Free memory.
        delete j1;
    }
}

/*=============================================================================
 * Private members
 *---------------------------------------------------------------------------*/

/**
 * @details
 * Returns a single, populated Jones scalar.
 */
void JonesTest::construct_double2_input(int i, double2& m)
{
    m = make_double2(i, i);
}

/**
 * @details
 * Returns a single, populated 2x2 Jones matrix.
 */
void JonesTest::construct_double4c_input(int i, double4c& m)
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
void JonesTest::construct_double4c_output_matrix_matrix(int i, int j,
        double4c& m)
{
    // Get the input matrix.
    double4c p,q;
    construct_double4c_input(i, p);
    construct_double4c_input(j, q);

    // Compute p*q.
    m.a = complex_add(complex_mul(p.a, q.a), complex_mul(p.b, q.c));
    m.b = complex_add(complex_mul(p.a, q.b), complex_mul(p.b, q.d));
    m.c = complex_add(complex_mul(p.c, q.a), complex_mul(p.d, q.c));
    m.d = complex_add(complex_mul(p.c, q.b), complex_mul(p.d, q.d));
}

/**
 * @details
 * Returns a single, populated 2x2 Jones matrix.
 */
void JonesTest::construct_double4c_output_matrix_scalar(int i, int j,
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
void JonesTest::construct_double4c_output_scalar_scalar(int i, int j,
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
void JonesTest::construct_float2_input(int i, float2& m)
{
    m = make_float2(i, i);
}

/**
 * @details
 * Returns a single, populated 2x2 Jones matrix.
 */
void JonesTest::construct_float4c_input(int i, float4c& m)
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
void JonesTest::construct_float4c_output_matrix_matrix(int i, int j,
        float4c& m)
{
    // Get the input matrix.
    float4c p,q;
    construct_float4c_input(i, p);
    construct_float4c_input(j, q);

    // Compute p*q.
    m.a = complex_add(complex_mul(p.a, q.a), complex_mul(p.b, q.c));
    m.b = complex_add(complex_mul(p.a, q.b), complex_mul(p.b, q.d));
    m.c = complex_add(complex_mul(p.c, q.a), complex_mul(p.d, q.c));
    m.d = complex_add(complex_mul(p.c, q.b), complex_mul(p.d, q.d));
}

/**
 * @details
 * Returns a single, populated 2x2 Jones matrix.
 */
void JonesTest::construct_float4c_output_matrix_scalar(int i, int j,
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
void JonesTest::construct_float4c_output_scalar_scalar(int i, int j,
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
oskar_Jones* JonesTest::construct_jones_host(int type, int n_src,
        int n_stat, int offset)
{
    oskar_Jones* m = new oskar_Jones(type, n_src, n_stat, 0);
    int n = n_src * n_stat;
    if (type == OSKAR_JONES_FLOAT_MATRIX)
    {
        float4c* ptr = (float4c*)m->data;
        for (int i = 0; i < n; ++i)
            construct_float4c_input(i + offset, ptr[i]);
    }
    else if (type == OSKAR_JONES_DOUBLE_MATRIX)
    {
        double4c* ptr = (double4c*)m->data;
        for (int i = 0; i < n; ++i)
            construct_double4c_input(i + offset, ptr[i]);
    }
    else if (type == OSKAR_JONES_FLOAT_SCALAR)
    {
        float2* ptr = (float2*)m->data;
        for (int i = 0; i < n; ++i)
            construct_float2_input(i + offset, ptr[i]);
    }
    else if (type == OSKAR_JONES_DOUBLE_SCALAR)
    {
        double2* ptr = (double2*)m->data;
        for (int i = 0; i < n; ++i)
            construct_double2_input(i + offset, ptr[i]);
    }

    // Return a pointer to the matrix structure.
    return m;
}

/**
 * @details
 * Returns a populated Jones matrix in device memory.
 */
oskar_Jones* JonesTest::construct_jones_device(int type, int n_src,
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
void JonesTest::check_matrix_matrix(const oskar_Jones* jones,
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
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("a.x" + oskar_to_std_string(i), t.a.x, ptr[i].a.x, abs(t.a.x * 5e-5));
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("a.y" + oskar_to_std_string(i), t.a.y, ptr[i].a.y, abs(t.a.y * 5e-5));
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("b.x" + oskar_to_std_string(i), t.b.x, ptr[i].b.x, abs(t.b.x * 5e-5));
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("b.y" + oskar_to_std_string(i), t.b.y, ptr[i].b.y, abs(t.b.y * 5e-5));
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("c.x" + oskar_to_std_string(i), t.c.x, ptr[i].c.x, abs(t.c.x * 5e-5));
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("c.y" + oskar_to_std_string(i), t.c.y, ptr[i].c.y, abs(t.c.y * 5e-5));
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("d.x" + oskar_to_std_string(i), t.d.x, ptr[i].d.x, abs(t.d.x * 5e-5));
            CPPUNIT_ASSERT_DOUBLES_EQUAL_MESSAGE("d.y" + oskar_to_std_string(i), t.d.y, ptr[i].d.y, abs(t.d.y * 5e-5));
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
void JonesTest::check_matrix_scalar(const oskar_Jones* data,
        int offset1, int offset2)
{

}

/**
 * @details
 * Checks result after performing scalar-scalar multiply.
 */
void JonesTest::check_scalar_scalar(const oskar_Jones* data,
        int offset1, int offset2)
{

}

/**
 * @details
 * Checks for errors and fails the test if one is found.
 */
void JonesTest::fail_on_error(int err)
{
    if (err > 0)
        CPPUNIT_FAIL("CUDA error, code " + oskar_to_std_string(err) +
                ": " + cudaGetErrorString((cudaError_t)err));
    else if (err < 0)
        CPPUNIT_FAIL("OSKAR error, code " + oskar_to_std_string(err));
}
