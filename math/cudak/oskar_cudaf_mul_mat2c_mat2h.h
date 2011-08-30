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

#ifndef OSKAR_CUDAF_MUL_MAT2C_MAT2H_H_
#define OSKAR_CUDAF_MUL_MAT2C_MAT2H_H_

/**
 * @file oskar_cudaf_mul_mat2c_mat2h.h
 */

#include "utility/oskar_cuda_eclipse.h"
#include "math/cudak/oskar_cudaf_mul_c_c.h"
#include "math/cudak/oskar_cudaf_mul_c_c_conj.h"

/**
 * @brief
 * CUDA device function to multiply a complex 2x2 matrix and a
 * Hermitian 2x2 matrix (single precision).
 *
 * @details
 * This inline device function multiplies together two complex 2x2 matrices,
 * where the second one is Hermitian.
 *
 * The second matrix is assumed to contain values as follows:
 *
 *   ( a   b )
 *   ( -   d )
 *
 * and a and d must both be real.
 *
 * Matrix multiplication is done in the order M1 = M1 * M2.
 *
 * @param[in,out] m1 On input, the first complex matrix; on output, the result.
 * @param[in]     m2 The second complex matrix.
 */
__device__ __forceinline__ void oskar_cudaf_mul_mat2c_mat2h_f(
        float4c& m1, const float4c& m2)
{
    // Before anything else, copy a and c from the input matrix.
    float2 a = m1.a;
    float2 c = m1.c;

    // Declare temporaries.
    float2 t;

    // First, evaluate result a.
    m1.a.x *= m2.a.x;
    m1.a.y *= m2.a.x;
    oskar_cudaf_mul_c_c_conj_f(m1.b, m2.b, t);
    m1.a.x += t.x;
    m1.a.y += t.y;

    // Second, evaluate result c.
    m1.c.x *= m2.a.x;
    m1.c.y *= m2.a.x;
    oskar_cudaf_mul_c_c_conj_f(m1.d, m2.b, t);
    m1.c.x += t.x;
    m1.c.y += t.y;

    // Third, evaluate result b.
    m1.b.x *= m2.d.x;
    m1.b.y *= m2.d.x;
    oskar_cudaf_mul_c_c_f(a, m2.b, t);
    m1.b.x += t.x;
    m1.b.y += t.y;

    // Fourth, evaluate result d.
    m1.d.x *= m2.d.x;
    m1.d.y *= m2.d.x;
    oskar_cudaf_mul_c_c_f(c, m2.b, t);
    m1.d.x += t.x;
    m1.d.y += t.y;
}

/**
 * @brief
 * CUDA device function to multiply a complex 2x2 matrix and a
 * Hermitian 2x2 matrix (double precision).
 *
 * @details
 * This inline device function multiplies together two complex 2x2 matrices,
 * where the second one is Hermitian.
 *
 * The second matrix is assumed to contain values as follows:
 *
 *   ( a   b )
 *   ( -   d )
 *
 * and a and d must both be real.
 *
 * Matrix multiplication is done in the order M1 = M1 * M2.
 *
 * @param[in,out] m1 On input, the first complex matrix; on output, the result.
 * @param[in]     m2 The second complex matrix.
 */
__device__ __forceinline__ void oskar_cudaf_mul_mat2c_mat2h_d(
        double4c& m1, const double4c& m2)
{
    // Before anything else, copy a and c from the input matrix.
    double2 a = m1.a;
    double2 c = m1.c;

    // Declare temporaries.
    double2 t;

    // First, evaluate result a.
    m1.a.x *= m2.a.x;
    m1.a.y *= m2.a.x;
    oskar_cudaf_mul_c_c_conj_d(m1.b, m2.b, t);
    m1.a.x += t.x;
    m1.a.y += t.y;

    // Second, evaluate result c.
    m1.c.x *= m2.a.x;
    m1.c.y *= m2.a.x;
    oskar_cudaf_mul_c_c_conj_d(m1.d, m2.b, t);
    m1.c.x += t.x;
    m1.c.y += t.y;

    // Third, evaluate result b.
    m1.b.x *= m2.d.x;
    m1.b.y *= m2.d.x;
    oskar_cudaf_mul_c_c_d(a, m2.b, t);
    m1.b.x += t.x;
    m1.b.y += t.y;

    // Fourth, evaluate result d.
    m1.d.x *= m2.d.x;
    m1.d.y *= m2.d.x;
    oskar_cudaf_mul_c_c_d(c, m2.b, t);
    m1.d.x += t.x;
    m1.d.y += t.y;
}

#endif // OSKAR_CUDAF_MUL_MAT2C_MAT2H_H_
