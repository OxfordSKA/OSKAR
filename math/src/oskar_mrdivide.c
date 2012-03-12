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


#include "math/oskar_mrdivide.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_free.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#define MAX( a, b ) ( ((a) > (b)) ? (a) : (b) )
#define MIN( a, b ) ( ((a) < (b)) ? (a) : (b) )

#ifndef OSKAR_NO_LAPACK
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
#endif



int oskar_mrdivide(oskar_Mem* C, const oskar_Mem* A, int rows_A, int cols_A,
        const oskar_Mem* B, int rows_B, int cols_B)
{
    return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;

//    int n, m, lda, ldb, nrhs, info, type, location, i, j;
//    oskar_Mem ipiv;
//    char trans;
//
//    if (A == NULL || B == NULL || C == NULL)
//        return OSKAR_ERR_INVALID_ARGUMENT;
//
//#ifdef OSKAR_NO_LAPACK
//    /* LAPACK not available ... :( */
//    return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
//#endif
//
//    if (A->location != OSKAR_LOCATION_CPU ||
//            B->location != OSKAR_LOCATION_CPU ||
//            C->location != OSKAR_LOCATION_CPU)
//    {
//        return OSKAR_ERR_BAD_LOCATION;
//    }
//
//    /* Common (double & single) */
//    n = rows_B;
//    m = cols_B;
//    lda = MAX(1, m);
//    oskar_mem_init(&ipiv, OSKAR_INT, OSKAR_LOCATION_CPU, MIN(m, n), OSKAR_TRUE);
//    info = 0;
//    trans = 'N';
//    ldb = MAX(1, n);
//    nrhs = 1;
//
//    /* Double precision */
//    if (A->type == OSKAR_DOUBLE && B->type == OSKAR_DOUBLE && C->type == OSKAR_DOUBLE)
//    {
//        /* Copy A into C */
//        for (j = 0; j < rows_A; ++j)
//        {
//            for (i = 0; i < cols_A; ++i)
//            {
//                ((double*)C->data)[j * cols_A + i] = ((double*)A->data)[j * cols_A + i];
//            }
//        }
//
//#ifndef OSKAR_NO_LAPACK
//        /* computes an LU factorisation of a general M-by-N matrix */
//        /* A = P * L * U */
//        /* http://www.netlib.org/lapack/double/dgetrf.f */
//        dgetrf_(&m, &n, (double*)B->data, &lda, (int*)ipiv.data, &info);
//#endif
//
//#ifndef OSKAR_NO_LAPACK
//        /* http://www.netlib.org/lapack/double/dgetrs.f */
//        /*  solves a system of linear equations */
//        /* A * X = B  or  A**T * X = B */
//        /*  with a general N-by-N matrix A using the LU factorisation computed
//         *  by DGETRF*/
//        dgetrs_(&trans, &n, &nrhs, (double*)B->data, &ldb, (int*)ipiv.data,
//                (double*)C->data, &lda, &info);
//#endif
//    }
//    /* Single precision */
//    else if (A->type == OSKAR_SINGLE && B->type == OSKAR_SINGLE && C->type == OSKAR_SINGLE)
//    {
//        return OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
//    }
//    else
//    {
//        return OSKAR_ERR_BAD_DATA_TYPE;
//    }
//
//    /* clean up */
//    oskar_mem_free(&ipiv);
//
//    return OSKAR_SUCCESS;
}


#ifdef __cplusplus
}
#endif
