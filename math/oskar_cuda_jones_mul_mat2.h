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

#ifndef OSKAR_CUDA_JONES_MUL_MAT2_H_
#define OSKAR_CUDA_JONES_MUL_MAT2_H_

/**
 * @file oskar_cuda_jones_mul_mat2.h
 */

#include "oskar_global.h"
#include "utility/oskar_vector_types.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Function to multiply together two Jones matrices (single precision).
 *
 * @details
 * This function multiplies together two complex Jones matrices to give a new
 * Jones matrix.
 *
 * The matrix multiplication is done in the order M = (J1 * J2).
 *
 * @param[in] n  The size of the input arrays.
 * @param[in] d_j1 Array of first input Jones matrices.
 * @param[in] d_j2 Array of second input Jones matrices.
 * @param[out] d_m Array of output Jones matrices.
 */
OSKAR_EXPORT
int oskar_cuda_jones_mul_mat2_f(int n, const float4c* d_j1,
        const float4c* d_j2, float4c* d_m);

/**
 * @brief
 * Function to multiply together two Jones matrices (double precision).
 *
 * @details
 * This function multiplies together two complex Jones matrices to give a new
 * Jones matrix.
 *
 * The matrix multiplication is done in the order M = (J1 * J2).
 *
 * @param[in] n  The size of the input arrays.
 * @param[in] d_j1 Array of first input Jones matrices.
 * @param[in] d_j2 Array of second input Jones matrices.
 * @param[out] d_m Array of output Jones matrices.
 */
OSKAR_EXPORT
int oskar_cuda_jones_mul_mat2_d(int n, const double4c* d_j1,
        const double4c* d_j2, double4c* d_m);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CUDA_JONES_MUL_MAT2_H_ */
