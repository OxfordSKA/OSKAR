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

#ifndef OSKAR_MATH_CORE_CTRIMAT_H_
#define OSKAR_MATH_CORE_CTRIMAT_H_

/**
 * @file oskar_math_core_ctrimat.h
 */

#include "oskar_math_windows.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Extracts the triangular half of a complex matrix (single precision).
 *
 * @details
 * This function extracts the triangular half of a complex square matrix.
 * The diagonals are excluded.
 *
 * @param[in] n The matrix dimension (matrix is n-by-n).
 * @param[in] a The matrix (length 2 * n * n).
 * @param[out] b The triangular half (length 2 * (n * (n - 1) / 2)).
 */
DllExport
void oskar_math_coref_ctrimat(int n, const float* a, float* b);

/**
 * @brief
 * Extracts the triangular half of a complex matrix (double precision).
 *
 * @details
 * This function extracts the triangular half of a complex square matrix.
 * The diagonals are excluded.
 *
 * @param[in] n The matrix dimension (matrix is n-by-n).
 * @param[in] a The matrix (length 2 * n * n).
 * @param[out] b The triangular half (length 2 * (n * (n - 1) / 2)).
 */
DllExport
void oskar_math_cored_ctrimat(int n, const double* a, double* b);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_MATH_CORE_CTRIMAT_H_
