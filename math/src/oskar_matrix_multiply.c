/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <oskar_matrix_multiply.h>

#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )

#ifdef __cplusplus
extern "C" {
#endif

enum CBLAS_ORDER {CblasRowMajor=101, CblasColMajor=102};
enum CBLAS_TRANSPOSE {CblasNoTrans=111, CblasTrans=112, CblasConjTrans=113};

/*
#ifndef OSKAR_NO_BLAS
extern void sgemm_(const char* TransA, const char* TransB, const int M,
        const int N, const int K, const float alpha, const float* A,
        const int lda, const float* B, const int lbd, const float beta,
        const float* C, const int ldc);

extern void dgemm_(const char* TransA, const char* TransB, const int M,
        const int N, const int K, const double alpha, const double* A,
        const int lda, const double* B, const int lbd, const double beta,
        const double* C, const int ldc);
#endif
*/

void cblas_sgemm(const enum CBLAS_ORDER Order,
        const enum CBLAS_TRANSPOSE TransA,
        const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
        const int K, const float alpha, const float *A,
        const int lda, const float *B, const int ldb,
        const float beta, float *C, const int ldc);

void cblas_dgemm(const enum CBLAS_ORDER Order,
        const enum CBLAS_TRANSPOSE TransA,
        const enum CBLAS_TRANSPOSE TransB, const int M, const int N,
        const int K, const double alpha, const double *A,
        const int lda, const double *B, const int ldb,
        const double beta, double *C, const int ldc);


/* Single precision. */
void oskar_matrix_multiply_f(float* C,
        int rows_A, int cols_A, int rows_B, int cols_B,
        int transA, int transB, const float* A, const float* B, int* status)
{
#ifdef OSKAR_NO_CBLAS
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    return;
#else
    int M, N, K, LDA, LDB, LDC;
    enum CBLAS_TRANSPOSE tA, tB;
    float alpha = 1.0f, beta = 0.0f;

    /* [ C = alpha * A * B + beta * C ] */
    M = (!transA) ? rows_A : cols_A;
    N = (!transB) ? cols_B : rows_B;
    K = (!transA) ? cols_A : rows_A;

    if (K != ((!transB)? rows_B : cols_B))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    tA = transA ? CblasTrans : CblasNoTrans;
    tB = transB ? CblasTrans : CblasNoTrans;

    LDA = (!transA) ? MAX(K, 1) : MAX(M, 1);
    LDB = (!transB) ? MAX(N, 1) : MAX(K, 1);
    LDC = MAX(N, 1);

    cblas_sgemm(CblasRowMajor, tA, tB, M, N, K, alpha, A,
            LDA, B, LDB, beta, C, LDC);
#endif
}


/* Double precision. */
void oskar_matrix_multiply_d(double* C,
        int rows_A, int cols_A, int rows_B, int cols_B,
        int transA, int transB, const double* A, const double* B, int* status)
{
#ifdef OSKAR_NO_CBLAS
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
    return;
#else
    int M, N, K, LDA, LDB, LDC;
    enum CBLAS_TRANSPOSE tA, tB;
    double alpha = 1.0, beta = 0.0;

    /* [ C = alpha * A * B + beta * C ] */
    M = (!transA) ? rows_A : cols_A;
    N = (!transB) ? cols_B : rows_B;
    K = (!transA) ? cols_A : rows_A;

    if (K != ((!transB)? rows_B : cols_B))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    tA = transA ? CblasTrans : CblasNoTrans;
    tB = transB ? CblasTrans : CblasNoTrans;

    LDA = (!transA) ? MAX(K, 1) : MAX(M, 1);
    LDB = (!transB) ? MAX(N, 1) : MAX(K, 1);
    LDC = MAX(N, 1);

    cblas_dgemm(CblasRowMajor, tA, tB, M, N, K, alpha, A,
            LDA, B, LDB, beta, C, LDC);
#endif
}

#ifdef __cplusplus
}
#endif
