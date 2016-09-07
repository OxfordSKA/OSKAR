/*
 * Copyright (c) 2016, The University of Oxford
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

#ifndef OSKAR_LAPACK_SUBSET_H_
#define OSKAR_LAPACK_SUBSET_H_

/**
 * @file oskar_lapack_subset.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EXPORT
void oskar_dgetrf(const int m, const int n, double *A, const int lda,
        int *ipiv, int *info);

OSKAR_EXPORT
void oskar_sgetrf(const int m, const int n, float *A, const int lda,
        int *ipiv, int *info);

OSKAR_EXPORT
void oskar_dgetrs(const char *trans, const int n, const int nrhs,
        double *A, const int lda, int *ipiv, double *B, const int ldb);

OSKAR_EXPORT
void oskar_sgetrs(const char *trans, const int n, const int nrhs,
        float *A, const int lda, int *ipiv, float *B, const int ldb);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_LAPACK_SUBSET_H_ */
