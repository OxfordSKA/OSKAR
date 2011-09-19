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

#ifndef OSKAR_CUDAK_JONES_MUL_MAT2_C2_H_
#define OSKAR_CUDAK_JONES_MUL_MAT2_C2_H_

/**
 * @file oskar_cudak_jones_mul_mat2_c2.h
 */

#include "oskar_global.h"
#include "utility/oskar_vector_types.h"

/**
 * @brief
 * CUDA kernel to multiply together two Jones matrices and two complex scalars
 * (single precision).
 *
 * @details
 * This kernel multiplies together two complex Jones matrices and two complex
 * scalars to give a new Jones matrix.
 *
 * The matrix multiplication is done in the order M = (J1 * J2) * (S1 * S2).
 *
 * @param[in] n  The size of the input arrays.
 * @param[in] j1 Array of first input Jones matrices.
 * @param[in] j2 Array of second input Jones matrices.
 * @param[in] s1 Array of first input scalars.
 * @param[in] s2 Array of second input scalars.
 * @param[out] m Array of output Jones matrices.
 */
__global__
void oskar_cudak_jones_mul_mat2_c2_f(int n, const float4c* j1,
        const float4c* j2, const float2* s1, const float2* s2, float4c* m);

/**
 * @brief
 * CUDA kernel to multiply together two Jones matrices and two complex scalars
 * (double precision).
 *
 * @details
 * This kernel multiplies together two complex Jones matrices and two complex
 * scalars to give a new Jones matrix.
 *
 * The matrix multiplication is done in the order M = (J1 * J2) * (S1 * S2).
 *
 * @param[in] n  The size of the input arrays.
 * @param[in] j1 Array of first input Jones matrices.
 * @param[in] j2 Array of second input Jones matrices.
 * @param[in] s1 Array of first input scalars.
 * @param[in] s2 Array of second input scalars.
 * @param[out] m Array of output Jones matrices.
 */
__global__
void oskar_cudak_jones_mul_mat2_c2_d(int n, const double4c* j1,
        const double4c* j2, const double2* s1, const double2* s2, double4c* m);

#endif // OSKAR_CUDAK_JONES_MUL_MAT2_C2_H_
