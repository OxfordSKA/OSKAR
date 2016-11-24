/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include "math/oskar_matrix_multiply.h"

#define SWAP_INT( a, b ) {int t; t = a; a = b; b = t;}

/* M is number of rows of A, and number of rows of C.
 * N is number of columns of B, and number of columns of C.
 * K is number of columns of A, and number of rows of B. */
#define MATRIX_MULTIPLY_MACRO \
        int M, N, K, LDA, LDB, i, j, l;                               \
        const int row_major = 1;                                      \
        M = !transpose_a ? rows_a : cols_a;                           \
        N = !transpose_b ? cols_b : rows_b;                           \
        K = !transpose_a ? cols_a : rows_a;                           \
        if (K != (!transpose_b ? rows_b : cols_b))                    \
        {                                                             \
            *status = OSKAR_ERR_DIMENSION_MISMATCH;                   \
            return;                                                   \
        }                                                             \
        if (row_major)                                                \
        {                                                             \
            A = b;                                                    \
            B = a;                                                    \
            SWAP_INT(transpose_a, transpose_b);                       \
            SWAP_INT(M, N);                                           \
        }                                                             \
        else                                                          \
        {                                                             \
            A = a;                                                    \
            B = b;                                                    \
        }                                                             \
        LDA = !transpose_a ? M : K;                                   \
        LDB = !transpose_b ? K : N;                                   \
        if (M == 0 || N == 0) return;                                 \
        if (!transpose_b)                                             \
        {                                                             \
            if (!transpose_a)                                         \
            {                                                         \
                for (j = 0; j < N; ++j)                               \
                {                                                     \
                    for (i = 0; i < M; ++i) c[i + j * M] = zero;      \
                    for (l = 0; l < K; ++l)                           \
                    {                                                 \
                        x = B[l + j * LDB];                           \
                        if (x != zero)                                \
                        {                                             \
                            for (i = 0; i < M; ++i)                   \
                            {                                         \
                                c[i + j * M] += x * A[i + l * LDA];   \
                            }                                         \
                        }                                             \
                    }                                                 \
                }                                                     \
                return;                                               \
            }                                                         \
            else                                                      \
            {                                                         \
                for (j = 0; j < N; ++j)                               \
                {                                                     \
                    for (i = 0; i < M; ++i)                           \
                    {                                                 \
                        x = zero;                                     \
                        for (l = 0; l < K; ++l)                       \
                        {                                             \
                            x += A[l + i * LDA] * B[l + j * LDB];     \
                        }                                             \
                        c[i + j * M] = x;                             \
                    }                                                 \
                }                                                     \
                return;                                               \
            }                                                         \
        }                                                             \
        else                                                          \
        {                                                             \
            if (!transpose_a)                                         \
            {                                                         \
                for (j = 0; j < N; ++j)                               \
                {                                                     \
                    for (i = 0; i < M; ++i) c[i + j * M] = zero;      \
                    for (l = 0; l < K; ++l)                           \
                    {                                                 \
                        x = B[j + l * LDB];                           \
                        if (x != zero)                                \
                        {                                             \
                            for (i = 0; i < M; ++i)                   \
                            {                                         \
                                c[i + j * M] += x * A[i + l * LDA];   \
                            }                                         \
                        }                                             \
                    }                                                 \
                }                                                     \
                return;                                               \
            }                                                         \
            else                                                      \
            {                                                         \
                for (j = 0; j < N; ++j)                               \
                {                                                     \
                    for (i = 0; i < M; ++i)                           \
                    {                                                 \
                        x = zero;                                     \
                        for (l = 0; l < K; ++l)                       \
                        {                                             \
                            x += A[l + i * LDA] * B[j + l * LDB];     \
                        }                                             \
                        c[i + j * M] = x;                             \
                    }                                                 \
                }                                                     \
                return;                                               \
            }                                                         \
        }


#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
void oskar_matrix_multiply_f(float* c, int rows_a, int cols_a,
        int rows_b, int cols_b, int transpose_a, int transpose_b,
        const float* a, const float* b, int* status)
{
    float x;
    const float *A, *B, zero = 0.f;

    MATRIX_MULTIPLY_MACRO
}

/* Double precision. */
void oskar_matrix_multiply_d(double* c, int rows_a, int cols_a,
        int rows_b, int cols_b, int transpose_a, int transpose_b,
        const double* a, const double* b, int* status)
{
    double x;
    const double *A, *B, zero = 0.;

    MATRIX_MULTIPLY_MACRO
}

#ifdef __cplusplus
}
#endif
