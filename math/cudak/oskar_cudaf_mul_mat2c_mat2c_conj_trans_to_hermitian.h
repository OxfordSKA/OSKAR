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

#ifndef OSKAR_CUDAF_MUL_MAT2C_MAT2C_CONJ_TRANS_TO_HERMITIAN_H_
#define OSKAR_CUDAF_MUL_MAT2C_MAT2C_CONJ_TRANS_TO_HERMITIAN_H_

/**
 * @file oskar_cudaf_mul_mat2c_mat2c_conj_trans_to_hermitian.h
 */

#include "utility/oskar_cuda_eclipse.h"
#include "math/cudak/oskar_cudaf_mul_c_c_conj.h"

/**
 * @brief
 * CUDA device function to multiply two complex 2x2 matrices, first taking the
 * conjugate transpose of the second, and only evaluating the non-zero terms
 * of the Hermitian result (single precision).
 *
 * @details
 * This inline device function multiplies together two complex 2x2 matrices.
 * The Hermitian conjugate of the second matrix is taken before the
 * multiplication, and the result is assumed to be Hermitian, so only the
 * relevant terms are evaluated.
 *
 * The output is of the form:
 *
 *   ( a  b )
 *   ( -  d )
 *
 * Matrix multiplication is done in the order M1 = M1 * M2^H.
 *
 * @param[in] m1 On input, the first complex matrix; on output, the result.
 * @param[in] m2 The second complex matrix.
 */
__device__ __forceinline__
void oskar_cudaf_mul_mat2c_mat2c_conj_trans_to_hermitian_f(
        float4c& m1, const float4c& m2)
{
    // Before anything else, copy a and c from the input matrix.
    float2 a = m1.a;
    float2 c = m1.c;

    // First, evaluate result a, using c1 as a temporary.
    oskar_cudaf_mul_c_c_conj_f(m1.a, m2.a);
    oskar_cudaf_mul_c_c_conj_f(m1.b, m2.b, m1.c);
    m1.a.x += m1.c.x;
    m1.a.y += m1.c.y;

    // Second, evaluate result b, using c1 as a temporary.
    oskar_cudaf_mul_c_c_conj_f(m1.b, m2.d);
    oskar_cudaf_mul_c_c_conj_f(a, m2.c, m1.c);
    m1.b.x += m1.c.x;
    m1.b.y += m1.c.y;

    // Third, evaluate result d, using c1 as a temporary.
    oskar_cudaf_mul_c_c_conj_f(m1.d, m2.d);
    oskar_cudaf_mul_c_c_conj_f(c, m2.c, m1.c);
    m1.d.x += m1.c.x;
    m1.d.y += m1.c.y;

    // Set m1.c to zero.
    m1.c.x = 0.0f;
    m1.c.y = 0.0f;
}

/**
 * @brief
 * CUDA device function to multiply two complex 2x2 matrices, first taking the
 * conjugate transpose of the second, and only evaluating the non-zero terms
 * of the Hermitian result (double precision).
 *
 * @details
 * This inline device function multiplies together two complex 2x2 matrices.
 * The Hermitian conjugate of the second matrix is taken before the
 * multiplication, and the result is assumed to be Hermitian, so only the
 * relevant terms are evaluated.
 *
 * The output is of the form:
 *
 *   ( a  b )
 *   ( -  d )
 *
 * Matrix multiplication is done in the order M1 = M1 * M2^H.
 *
 * @param[in] m1 On input, the first complex matrix; on output, the result.
 * @param[in] m2 The second complex matrix.
 */
__device__ __forceinline__
void oskar_cudaf_mul_mat2c_mat2c_conj_trans_to_hermitian_d(
        double4c& m1, const double4c& m2)
{
    // Before anything else, copy a and c from the input matrix.
    double2 a = m1.a;
    double2 c = m1.c;

    // First, evaluate result a, using c1 as a temporary.
    oskar_cudaf_mul_c_c_conj_d(m1.a, m2.a);
    oskar_cudaf_mul_c_c_conj_d(m1.b, m2.b, m1.c);
    m1.a.x += m1.c.x;
    m1.a.y += m1.c.y;

    // Second, evaluate result b, using c1 as a temporary.
    oskar_cudaf_mul_c_c_conj_d(m1.b, m2.d);
    oskar_cudaf_mul_c_c_conj_d(a, m2.c, m1.c);
    m1.b.x += m1.c.x;
    m1.b.y += m1.c.y;

    // Third, evaluate result d, using c1 as a temporary.
    oskar_cudaf_mul_c_c_conj_d(m1.d, m2.d);
    oskar_cudaf_mul_c_c_conj_d(c, m2.c, m1.c);
    m1.d.x += m1.c.x;
    m1.d.y += m1.c.y;

    // Set m1.c to zero.
    m1.c.x = 0.0;
    m1.c.y = 0.0;
}

#endif // OSKAR_CUDAF_MUL_MAT2C_MAT2C_CONJ_TRANS_TO_HERMITIAN_H_
