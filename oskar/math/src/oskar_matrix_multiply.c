/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "math/oskar_matrix_multiply.h"

#define SWAP_INT( a, b ) {int t; t = a; a = b; b = t;}

/* M is number of rows of A, and number of rows of C.
 * N is number of columns of B, and number of columns of C.
 * K is number of columns of A, and number of rows of B. */
#define MATRIX_MULTIPLY_MACRO \
        int M = 0, N = 0, K = 0, LDA = 0, LDB = 0, i = 0, j = 0, l = 0;\
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
    float x = 0.0f;
    const float *A = 0, *B = 0, zero = 0.f;

    MATRIX_MULTIPLY_MACRO
}

/* Double precision. */
void oskar_matrix_multiply_d(double* c, int rows_a, int cols_a,
        int rows_b, int cols_b, int transpose_a, int transpose_b,
        const double* a, const double* b, int* status)
{
    double x = 0.0;
    const double *A = 0, *B = 0, zero = 0.;

    MATRIX_MULTIPLY_MACRO
}

#ifdef __cplusplus
}
#endif
