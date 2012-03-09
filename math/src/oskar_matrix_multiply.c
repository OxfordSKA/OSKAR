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


#include "math/oskar_matrix_multiply.h"
#include <stdlib.h>
#include <stdio.h>

#ifndef OSKAR_NO_CBLAS
#include <cblas.h>
#endif

#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )

#ifdef __cplusplus
extern "C" {
#endif

int oskar_matrix_multiply(oskar_Mem* C, int M, int N, int K,
        int transA, int transB, const oskar_Mem* A, const oskar_Mem* B)
{
    if (A == NULL || B == NULL || C == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;


    if (A->location == OSKAR_LOCATION_CPU &&
            B->location == OSKAR_LOCATION_CPU &&
            C->location == OSKAR_LOCATION_CPU)
    {
        if (A->type == OSKAR_DOUBLE && B->type == OSKAR_DOUBLE &&
                C->type == OSKAR_DOUBLE)
        {
#ifndef OSKAR_NO_CBLAS
            /* [ C = alpha * A * B + beta * C ] */

            int tA, tB;
            int lda, ldb, ldc;
            double alpha = 1.0, beta = 0.0;

            tA = transA ? CblasTrans : CblasNoTrans;
            tB = transB ? CblasTrans : CblasNoTrans;

            lda = (transA) ? MAX(M, 1) : MAX(K, 1);
            ldb = (transB) ? MAX(K, 1) : MAX(N, 1);
            ldc = MAX(N, 1);

            printf("trans a,b = %s, %s\n", transA?"true":"false", transB?"true":"false");
            printf("M, N, K: %i %i %i\n", M, N, K);
            printf("ld (a,b,c): %i %i %i\n", lda, ldb, ldc);

            cblas_dgemm(CblasRowMajor, tA, tB, M, N, K, alpha, (double*)A->data,
                    lda, (double*)B->data, ldb, beta, (double*)C->data, ldc);
#else
            /* TODO implement replacement for the blas funciton */
            return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
#endif
        }
        else if (A->type == OSKAR_SINGLE && B->type == OSKAR_SINGLE &&
                C->type == OSKAR_SINGLE)
        {
#ifndef OSKAR_NO_CBLAS
            return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
/*            int tA, tB;
            int lda, ldb, ldc;
            float alpha = 1.0f, beta = 0.0f;

            tA = (transA ? CblasTrans : CblasNoTrans);
            tB = (transB ? CblasTrans : CblasNoTrans);

            lda = (transA) ? MAX(1, K) : MAX(1, M);
            ldb = (transB) ? MAX(1, N) : MAX(1, K);
            ldc = MAX(1, M);

            cblas_sgemm(CblasRowMajor, tA, tB, M, N, K, alpha, (float*)A->data,
                    lda, (float*)B->data, ldb, beta, (float*)C->data, ldc);
                    */
#else
            /* TODO implement replacement for the blas funciton */
            return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
#endif
        }
        else
        {
            return OSKAR_ERR_BAD_DATA_TYPE;
        }
    }
    else
    {
        return OSKAR_ERR_BAD_LOCATION;
    }

    return OSKAR_SUCCESS;
}


#ifdef __cplusplus
}
#endif
