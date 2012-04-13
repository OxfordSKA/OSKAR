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

#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )

#ifndef OSKAR_NO_LAPACK
extern "C" {
/* http://www.netlib.org/lapack/double/dgels.f */
extern void dgels_(const char* trans, const int* m, const int* n,
        const int* nrhs, double* A, const int *lda, double* b, const int* ldb,
        double* work, const int* lwork, int* info);

/* http://www.netlib.org/lapack/double/dgetrf.f */
extern void dgetrf_(const int* m, const int* n, double* A, const int* lda,
        int* ipiv, int* info);

/* http://www.netlib.org/lapack/double/dgetrs.f */
extern void dgetrs_(const char* trans, const int* n, const int* nrhs,
        double* A, const int* lda, int* ipiv, double* B, const int* lba,
        int* info);
}
#endif

void Test_matrix_math::test_multiply()
{
    int type = OSKAR_DOUBLE;
    int location = OSKAR_LOCATION_CPU;

    int cols_A = 3;
    int rows_A = 4;

    int cols_B = 2;
    int rows_B = 3;

    oskar_Mem A(type, location, cols_A * rows_A);
    oskar_Mem B(type, location, cols_B * rows_B);

    double* a = (double*)A.data;
    // 14 9 3
    // 2 11 15
    // 0 12 17
    // 5 2 3
    a[0] = 14;
    a[1] = 9;
    a[2] = 3;
    a[3] = 2;
    a[4] = 11;
    a[5] = 15;
    a[6] = 0;
    a[7] = 12;
    a[8] = 17;
    a[9] = 5;
    a[10] = 2;
    a[11] = 3;


//    double* b = (double*)B.data;
//    // 12 25
//    // 9 10
//    // 8 5
//    b[0] = 12;
//    b[1] = 25;
//    b[2] = 9;
//    b[3] = 10;
//    b[4] = 8;
//    b[5] = 5;
    cols_B = cols_A;
    rows_B = rows_A;

    int transA = OSKAR_FALSE;
    int transB = OSKAR_TRUE;

    int rows_C = (!transA) ? rows_A : cols_A;
    int cols_C = (!transB) ? cols_B : rows_B;

    oskar_Mem C(type, location, rows_C * cols_C);

#if !defined (OSKAR_NO_CBLAS)
    int err = oskar_matrix_multiply(&C,
            rows_A, cols_A, rows_A, cols_A,
            transA, transB, &A, &A);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), (int)OSKAR_SUCCESS,
            err);
#endif

//    printf("\nA:\n");
//    for (int j = 0; j < rows_A; ++j)
//    {
//        for (int i = 0; i < cols_A; ++i)
//        {
//            printf("% -.4f ", a[j * cols_A + i]);
//        }
//        printf("\n");
//    }
//    printf("\nB:\n");
//    for (int j = 0; j < rows_B; ++j)
//    {
//        for (int i = 0; i < cols_B; ++i)
//        {
//            printf("% -.4f ", b[j * cols_B + i]);
//        }
//        printf("\n");
//    }

//    double* c = (double*)C.data;
//    printf("\nC:\n");
//    for (int j = 0; j < rows_C; ++j)
//    {
//        for (int i = 0; i < cols_C; ++i)
//        {
//            printf("% -.4f ", c[j * cols_C + i]);
//        }
//        printf("\n");
//    }
}


void Test_matrix_math::dgels_test()
{
#if defined (OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK)
    return;
#endif


    int rows_A = 3;
    int cols_A = 3;
    oskar_Mem A(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, rows_A * cols_A);
    ((double*)A.data)[0] = 14; ((double*)A.data)[1] = 9;  ((double*)A.data)[2] = 3;
    ((double*)A.data)[3] = 2;  ((double*)A.data)[4] = 11; ((double*)A.data)[5] = 15;
    ((double*)A.data)[6] = 0;  ((double*)A.data)[7] = 12; ((double*)A.data)[8] = 17;

    int cols_B = 3;
    oskar_Mem B(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, cols_B);
    ((double*)B.data)[0] = 16;
    ((double*)B.data)[1] = 32;
    ((double*)B.data)[2] = 35;

//    printf("\nA:\n");
//    for (int j = 0; j < rows_A; ++j)
//    {
//        for (int i = 0; i < cols_A; ++i)
//        {
//            printf("%f ", ((double*)A.data)[j * cols_A + i]);
//        }
//        printf("\n");
//    }

    // denom = inv(A'*A);
    oskar_Mem denom(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, rows_A * cols_A);
    int err = oskar_matrix_multiply(&denom, rows_A, cols_A, rows_A, cols_A,
            OSKAR_TRUE, OSKAR_FALSE, &A, &A);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), (int)OSKAR_SUCCESS, err);

//    printf("\nsum(A):\n");
//    for (int j = 0; j < cols_B; ++j)
//    {
//        printf("%f ", ((double*)B.data)[j]);
//    }
//    printf("\n");

    char trans = 'N';
    int m = rows_A;
    int n = cols_A;
    int nrhs = 1;
    int lda = MAX(1, m);
    int ldb = MAX(m, n);

    int MN = MIN(m, n);
    int lwork = MAX(1, MN + MAX(MN, nrhs) * 20);
    oskar_Mem work(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, lwork);
    int info = 0;

//    printf("- lwork = %i\n", lwork);

    dgels_(&trans, &m, &n, &nrhs, (double*)(denom.data), &lda,
            (double*)(B.data), &ldb, (double*)work.data, &lwork, &info);

//    printf("\n");
//    printf("- info    = %i\n", info);
//    printf("- work[0] = %f\n", ((double*)work.data)[0]);
//
//    printf("\nresult:\n");
//    for (int j = 0; j < cols_B; ++j)
//    {
//        printf("%f ", ((double*)B.data)[j]);
//    }
//    printf("\n");


}


void Test_matrix_math::dgetrs_test()
{
#if defined (OSKAR_NO_CBLAS) || defined(OSKAR_NO_LAPACK)
    return;
#endif

    int rows_A = 3;
    int cols_A = 3;
    oskar_Mem A(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, rows_A * cols_A);
    ((double*)A.data)[0] = 14; ((double*)A.data)[1] = 9;  ((double*)A.data)[2] = 3;
    ((double*)A.data)[3] = 2;  ((double*)A.data)[4] = 11; ((double*)A.data)[5] = 15;
    ((double*)A.data)[6] = 0;  ((double*)A.data)[7] = 12; ((double*)A.data)[8] = 17;

    int cols_B = 3;
    oskar_Mem B(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, cols_B);
    ((double*)B.data)[0] = 16;
    ((double*)B.data)[1] = 32;
    ((double*)B.data)[2] = 35;

    // denom = inv(A'*A);
    oskar_Mem denom(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, rows_A * cols_A);

    int err = oskar_matrix_multiply(&denom, rows_A, cols_A, rows_A, cols_A,
            OSKAR_TRUE, OSKAR_FALSE, &A, &A);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), (int)OSKAR_SUCCESS, err);

    int n = rows_A;
    int m = cols_A;
    int lda = MAX(1, m);
    oskar_Mem ipiv(OSKAR_INT, OSKAR_LOCATION_CPU, MIN(m,n));
    int info = 0;

    dgetrf_(&m, &n, (double*)denom.data, &lda, (int*)ipiv.data, &info);

    char trans = 'N';
    int ldb = MAX(1, n);
    int nrhs = 1;

    dgetrs_(&trans, &n, &nrhs, (double*)denom.data, &lda, (int*)ipiv.data,
            (double*)B.data, &ldb, &info);

//    printf("\nresult:\n");
//    for (int j = 0; j < cols_B; ++j)
//    {
//        printf("%f ", ((double*)B.data)[j]);
//    }
//    printf("\n");

}

void Test_matrix_math::sumX_div_XX()
{
//    int rows_X = 3;
//    int cols_X = 3;
//    oskar_Mem X(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, rows_X * cols_X);
//    ((double*)X.data)[0] = 14; ((double*)X.data)[1] = 9;  ((double*)X.data)[2] = 3;
//    ((double*)X.data)[3] = 2;  ((double*)X.data)[4] = 11; ((double*)X.data)[5] = 15;
//    ((double*)X.data)[6] = 0;  ((double*)X.data)[7] = 12; ((double*)X.data)[8] = 17;
//
//    oskar_Mem sumX(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, cols_X);
//    for (int j = 0; j < rows_X; ++j)
//    {
//        for (int i = 0; i < cols_X; ++i)
//        {
//            ((double*)sumX.data)[i] += ((double*)X.data)[j * cols_X + i];
//        }
//    }
//
//    oskar_Mem XX(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, rows_X * cols_X);
//    int err = oskar_matrix_multiply(&XX, rows_X, cols_X, rows_X, cols_X,
//            OSKAR_TRUE, OSKAR_FALSE, &X, &X);
//    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), (int)OSKAR_SUCCESS, err);
//
//    oskar_Mem result(OSKAR_DOUBLE, OSKAR_LOCATION_CPU, cols_X);
//    err = oskar_mrdivide(&result, &sumX, 1, cols_X, &XX, rows_X, cols_X);
//    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), (int)OSKAR_SUCCESS, err);
//
//    printf("\nresult:\n");
//    for (int j = 0; j < cols_X; ++j)
//    {
//        printf("%f ", ((double*)result.data)[j]);
//    }
//    printf("\n");
}

