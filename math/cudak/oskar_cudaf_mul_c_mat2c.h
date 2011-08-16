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

#ifndef OSKAR_CUDAF_MUL_C_MAT2C_H_
#define OSKAR_CUDAF_MUL_C_MAT2C_H_

/**
 * @file oskar_cudaf_mul_c_mat2c.h
 */

#include "utility/oskar_cuda_eclipse.h"
#include "math/cudak/oskar_cudaf_mul_c_c.h"

/**
 * @brief
 * CUDA device function to multiply a complex 2x2 matrix and a complex scalar
 * (single precision).
 *
 * @details
 * This inline device function multiplies together a complex scalar number
 * and a complex 2x2 matrix to give a new complex 2x2 matrix.
 *
 * @param[in] a The complex scalar number.
 * @param[in] m The complex 2x2 matrix.
 * @param[out] out The output complex 2x2 matrix.
 */
__device__ __forceinline__ void oskar_cudaf_mul_c_mat2c_f(const float2& a,
        const float4c& m, float4c& out)
{
    // Multiply matrix by complex scalar.
    float2 t;
    oskar_cudaf_mul_c_c_f(m.a, a, t);
    out.a = t;
    oskar_cudaf_mul_c_c_f(m.b, a, t);
    out.b = t;
    oskar_cudaf_mul_c_c_f(m.c, a, t);
    out.c = t;
    oskar_cudaf_mul_c_c_f(m.d, a, t);
    out.d = t;
}

/**
 * @brief
 * CUDA device function to multiply a complex 2x2 matrix and a complex scalar
 * (double precision).
 *
 * @details
 * This inline device function multiplies together a complex scalar number
 * and a complex 2x2 matrix to give a new complex 2x2 matrix.
 *
 * @param[in] a The complex scalar number.
 * @param[in] m The complex 2x2 matrix.
 * @param[out] out The output complex 2x2 matrix.
 */
__device__ __forceinline__ void oskar_cudaf_mul_c_mat2c_d(const double2& a,
        const double4c& m, double4c& out)
{
    // Multiply matrix by complex scalar.
    double2 t;
    oskar_cudaf_mul_c_c_d(m.a, a, t);
    out.a = t;
    oskar_cudaf_mul_c_c_d(m.b, a, t);
    out.b = t;
    oskar_cudaf_mul_c_c_d(m.c, a, t);
    out.c = t;
    oskar_cudaf_mul_c_c_d(m.d, a, t);
    out.d = t;
}

#endif // OSKAR_CUDAF_MUL_C_MAT2C_H_
