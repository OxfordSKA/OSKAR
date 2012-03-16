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

int oskar_matrix_multiply(oskar_Mem* C,
        int rows_A, int cols_A, int rows_B, int cols_B,
        int transA, int transB, const oskar_Mem* A, const oskar_Mem* B)
{
    int M, N, K;
    int LDA, LDB, LDC;

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
            double alpha = 1.0, beta = 0.0;
            M = (!transA) ? rows_A : cols_A;
            N = (!transB) ? cols_B : rows_B;
            K = (!transA) ? cols_A : rows_A;

            if (K != ((!transB)? rows_B : cols_B))
            {
                return OSKAR_ERR_DIMENSION_MISMATCH;
            }

            tA = transA ? CblasTrans : CblasNoTrans;
            tB = transB ? CblasTrans : CblasNoTrans;

            LDA = (!transA) ? MAX(K, 1) : MAX(M, 1);
            LDB = (!transB) ? MAX(N, 1) : MAX(K, 1);
            LDC = MAX(N, 1);

            cblas_dgemm(CblasRowMajor, tA, tB, M, N, K, alpha,
                    (double*)A->data,
                    LDA, (double*)B->data, LDB, beta, (double*)C->data, LDC);
#else
            /* TODO implement replacement for the BLAS function */
            return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
#endif
        }
        else if (A->type == OSKAR_SINGLE && B->type == OSKAR_SINGLE &&
                C->type == OSKAR_SINGLE)
        {
#ifndef OSKAR_NO_CBLAS
            return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
#else
            /* TODO implement replacement for the BLAS function */
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