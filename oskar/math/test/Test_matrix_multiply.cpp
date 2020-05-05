/*
 * Copyright (c) 2015, The University of Oxford
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

#include "math/oskar_matrix_multiply.h"
#include "math/define_multiply.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_vector_types.h"

#include <cstdlib>

TEST(matrix_multiply, test_order)
{
    int status = 0;
    double tol = 1e-12;
    double a[] = {
            1., 2., 3., 4.,
            5., 6., 7., 8.
    };
    double b[] = {
            1.1, 1.2,
            2.3, 2.4,
            3.5, 3.6,
            4.7, 4.8
    };
    double r_ab[4], r_ba[16];
    oskar_matrix_multiply_d(r_ab, 2, 4, 4, 2, 0, 0, a, b, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_NEAR(r_ab[0], 35.0, tol);
    EXPECT_NEAR(r_ab[1], 36.0, tol);
    EXPECT_NEAR(r_ab[2], 81.4, tol);
    EXPECT_NEAR(r_ab[3], 84.0, tol);

    oskar_matrix_multiply_d(r_ba, 4, 2, 2, 4, 0, 0, b, a, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_NEAR(r_ba[0], 7.1, tol);
    EXPECT_NEAR(r_ba[1], 9.4, tol);
    EXPECT_NEAR(r_ba[2], 11.7, tol);
    EXPECT_NEAR(r_ba[3], 14.0, tol);
    EXPECT_NEAR(r_ba[4], 14.3, tol);
    EXPECT_NEAR(r_ba[5], 19.0, tol);
    EXPECT_NEAR(r_ba[6], 23.7, tol);
    EXPECT_NEAR(r_ba[7], 28.4, tol);
    EXPECT_NEAR(r_ba[8], 21.5, tol);
    EXPECT_NEAR(r_ba[9], 28.6, tol);
    EXPECT_NEAR(r_ba[10], 35.7, tol);
    EXPECT_NEAR(r_ba[11], 42.8, tol);
    EXPECT_NEAR(r_ba[12], 28.7, tol);
    EXPECT_NEAR(r_ba[13], 38.2, tol);
    EXPECT_NEAR(r_ba[14], 47.7, tol);
    EXPECT_NEAR(r_ba[15], 57.2, tol);
}

TEST(matrix_multiply, test_transpose_A)
{
    int status = 0;
    double tol = 1e-12;
    double m[] = {
            1.1, 1.2, 1.3,
            2.3, 2.4, 2.5,
            3.5, 3.6, 3.7,
            4.7, 4.8, 4.9
    };
    double r[9];

    // trans(M) * M: Should be OK.
    status = 0;
    oskar_matrix_multiply_d(r, 4, 3, 4, 3, 1, 0, m, m, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_NEAR(r[0], 40.84, tol);
    EXPECT_NEAR(r[1], 42.0, tol);
    EXPECT_NEAR(r[2], 43.16, tol);
    EXPECT_NEAR(r[3], 42.0, tol);
    EXPECT_NEAR(r[4], 43.2, tol);
    EXPECT_NEAR(r[5], 44.4, tol);
    EXPECT_NEAR(r[6], 43.16, tol);
    EXPECT_NEAR(r[7], 44.4, tol);
    EXPECT_NEAR(r[8], 45.64, tol);
}

TEST(matrix_multiply, test_transpose_B)
{
    int status = 0;
    double tol = 1e-12;
    double a[] = {
            1., 2., 3.,
            4., 5., 6.
    };
    double b[] = {
            1.1, 1.2, 1.3,
            2.3, 2.4, 2.5,
            3.5, 3.6, 3.7,
            4.7, 4.8, 4.9
    };
    double r[8];

    // A * trans(B): Should be OK.
    oskar_matrix_multiply_d(r, 2, 3, 4, 3, 0, 1, a, b, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_NEAR(r[0], 7.4, tol);
    EXPECT_NEAR(r[1], 14.6, tol);
    EXPECT_NEAR(r[2], 21.8, tol);
    EXPECT_NEAR(r[3], 29.0, tol);
    EXPECT_NEAR(r[4], 18.2, tol);
    EXPECT_NEAR(r[5], 36.2, tol);
    EXPECT_NEAR(r[6], 54.2, tol);
    EXPECT_NEAR(r[7], 72.2, tol);

    // A * B: Should fail.
    status = 0;
    oskar_matrix_multiply_d(r, 2, 3, 4, 3, 0, 0, a, b, &status);
    ASSERT_EQ(OSKAR_ERR_DIMENSION_MISMATCH, status);

    // B * A: Should fail.
    status = 0;
    oskar_matrix_multiply_d(r, 4, 3, 2, 3, 0, 0, b, a, &status);
    ASSERT_EQ(OSKAR_ERR_DIMENSION_MISMATCH, status);

    // trans(A) * B: Should fail.
    status = 0;
    oskar_matrix_multiply_d(r, 2, 3, 4, 3, 1, 0, a, b, &status);
    ASSERT_EQ(OSKAR_ERR_DIMENSION_MISMATCH, status);
}

TEST(matrix_multiply, matrix2x2)
{
    const double tol = 1e-6;
    double4c a, b;
    a.a.x = 0.1;
    a.a.y = 0.2;
    a.b.x = -0.3;
    a.b.y = 0.4;
    a.c.x = -0.5;
    a.c.y = -0.6;
    a.d.x = 0.7;
    a.d.y = -0.8;
    b = a;
    OSKAR_MUL_COMPLEX_MATRIX_CONJUGATE_TRANSPOSE_IN_PLACE(double2, a, b);
    EXPECT_NEAR(a.a.x, 0.3, tol);
    EXPECT_NEAR(a.a.y, 0.0, tol);
    EXPECT_NEAR(a.b.x, -0.7, tol);
    EXPECT_NEAR(a.b.y, 0.0, tol);
    EXPECT_NEAR(a.c.x, -0.7, tol);
    EXPECT_NEAR(a.c.y, 0.0, tol);
    EXPECT_NEAR(a.d.x, 1.74, tol);
    EXPECT_NEAR(a.d.y, 0.0, tol);
}
