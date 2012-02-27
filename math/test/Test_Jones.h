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

#ifndef TEST_JONES_H_
#define TEST_JONES_H_

/**
 * @file Test_Jones.h
 */

#include <cppunit/extensions/HelperMacros.h>

// Forward declarations.
struct oskar_Jones;
struct float2;
struct float4c;
struct double2;
struct double4c;
struct oskar_CudaDeviceInfo;

/**
 * @brief Unit test class that uses CppUnit.
 *
 * @details
 * This class uses the CppUnit testing framework to perform unit tests
 * on the class it is named after.
 */
class Test_Jones : public CppUnit::TestFixture
{
    public:
        CPPUNIT_TEST_SUITE(Test_Jones);
        CPPUNIT_TEST(test_join_inline_mat_mat_device);
        CPPUNIT_TEST(test_join_inline_mat_mat_device_host);
        CPPUNIT_TEST(test_join_inline_mat_mat_host);
        CPPUNIT_TEST(test_join_inline_mat_sca_device);
        CPPUNIT_TEST(test_join_inline_mat_sca_device_host);
        CPPUNIT_TEST(test_join_inline_mat_sca_host);
        CPPUNIT_TEST(test_join_inline_sca_sca_device);
        CPPUNIT_TEST(test_join_inline_sca_sca_device_host);
        CPPUNIT_TEST(test_join_inline_sca_sca_host);
        CPPUNIT_TEST(test_set_ones_device);
        CPPUNIT_TEST(test_set_ones_host);
//        CPPUNIT_TEST(test_performance);
        CPPUNIT_TEST_SUITE_END();

    public:
        Test_Jones();
        ~Test_Jones();

    public:
        // Test methods.
        void test_join_inline_mat_mat_device();
        void test_join_inline_mat_mat_device_host();
        void test_join_inline_mat_mat_host();
        void test_join_inline_mat_sca_device();
        void test_join_inline_mat_sca_device_host();
        void test_join_inline_mat_sca_host();
        void test_join_inline_sca_sca_device();
        void test_join_inline_sca_sca_device_host();
        void test_join_inline_sca_sca_host();
        void test_set_ones_device();
        void test_set_ones_host();
        void test_performance();

    private:
        void construct_double2_input(int i, double2& m);
        void construct_double2_output_scalar_scalar(int i, int j, double2& m);
        void construct_double4c_input(int i, double4c& m);
        void construct_double4c_output_matrix_matrix(int i, int j, double4c& m);
        void construct_double4c_output_matrix_scalar(int i, int j, double4c& m);
        void construct_float2_input(int i, float2& m);
        void construct_float2_output_scalar_scalar(int i, int j, float2& m);
        void construct_float4c_input(int i, float4c& m);
        void construct_float4c_output_matrix_matrix(int i, int j, float4c& m);
        void construct_float4c_output_matrix_scalar(int i, int j, float4c& m);
        oskar_Jones* construct_jones_host(int type, int n_stat, int n_src,
                int offset);
        oskar_Jones* construct_jones_device(int type, int n_stat, int n_src,
                int offset);
        void check_matrix_matrix(const oskar_Jones* data,
                int offset1, int offset2);
        void check_matrix_scalar(const oskar_Jones* data,
                int offset1, int offset2);
        void check_scalar_scalar(const oskar_Jones* data,
                int offset1, int offset2);
        void fail_on_error(int err);

    private:
        oskar_CudaDeviceInfo* device_;
};

/*=============================================================================
 * Helper functions
 *---------------------------------------------------------------------------*/

/**
 * @details
 * Converts the parameter to a C++ string.
 */
#include <sstream>

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

// Register the test class.
CPPUNIT_TEST_SUITE_REGISTRATION(Test_Jones);

#endif // TEST_JONES_H_
