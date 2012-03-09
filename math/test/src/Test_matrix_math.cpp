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

#include "math/test/Test_matrix_math.h"
#include "math/oskar_matrix_multiply.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_get_error_string.h"

#include <cmath>
#include <cstdio>


void Test_matrix_math::test_multiply()
{
    int type = OSKAR_DOUBLE;
    int location = OSKAR_LOCATION_CPU;

    int cols_A = 2;
    int rows_A = 3;

    int cols_B = 3;
    int rows_B = 2;

    oskar_Mem A(type, location, cols_A * rows_A);
    oskar_Mem B(type, location, cols_B * rows_B);

    double* a = (double*)A.data;
    // 1 2
    // 3 4
    // 5 6
    a[0] = 1.0;
    a[1] = 2.0;
    a[2] = 3.0;
    a[3] = 4.0;
    a[4] = 5.0;
    a[5] = 6.0;

    double* b = (double*)B.data;
    // 7  8  9
    // 10 11 12
    b[0] = 7.0;
    b[1] = 8.0;
    b[2] = 9.0;
    b[3] = 10.0;
    b[4] = 11.0;
    b[5] = 12.0;

    int transA = OSKAR_FALSE;
    int transB = OSKAR_FALSE;

    int M = (!transA) ? rows_A : cols_A; // rows of op(A) and C
    int N = (!transB) ? cols_B : rows_B; // columns of op(B) and C
    int K = (!transA) ? cols_A : rows_A; // columns of op(A) and rows of op(B)

    oskar_Mem C(type, location, M * N);

    printf("\nA:\n");
    for (int j = 0; j < rows_A; ++j)
    {
        for (int i = 0; i < cols_A; ++i)
        {
            printf("% -.4f ", a[j * cols_A + i]);
        }
        printf("\n");
    }
    printf("\nB:\n");
    for (int j = 0; j < rows_B; ++j)
    {
        for (int i = 0; i < cols_B; ++i)
        {
            printf("% -.4f ", b[j * cols_B + i]);
        }
        printf("\n");
    }

    int err = oskar_matrix_multiply(&C, M, N, K, transA, transB, &A, &B);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), (int)OSKAR_SUCCESS,
            err);

    double* c = (double*)C.data;
    printf("\nC:\n");
    for (int j = 0; j < M; ++j)
    {
        for (int i = 0; i < N; ++i)
        {
            printf("% -.4f ", c[j * N + i]);
        }
        printf("\n");
    }
}
