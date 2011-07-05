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

#ifndef OSKAR_CUDA_MATH_MATRIX2C_H_
#define OSKAR_CUDA_MATH_MATRIX2C_H_

#include "vector_types.h"

/**
 * @file oskar_cuda_math_matrix2c.h
 */

extern "C" {

/**
 * @brief
 * Complex 2x2 matrix (single precision).
 *
 * @details
 * Structure used to hold data for a complex 2x2 matrix.
 * The elements are arranged as:
 *
 *   ( a  b )
 *   ( c  d )
 */
struct __align__(16) Matrix2cf
{
    float2 a;
    float2 b;
    float2 c;
    float2 d;
};

/**
 * @brief
 * Complex 2x2 matrix (double precision).
 *
 * @details
 * Structure used to hold data for a complex 2x2 matrix.
 * The elements are arranged as:
 *
 *   ( a  b )
 *   ( c  d )
 */
struct __align__(16) Matrix2cd
{
    double2 a;
    double2 b;
    double2 c;
    double2 d;
};

}

#endif // OSKAR_CUDA_MATH_MATRIX2C_H_
