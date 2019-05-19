/*
 * Copyright (c) 2011-2019, The University of Oxford.
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

#ifndef OSKAR_VECTOR_TYPES_H_
#define OSKAR_VECTOR_TYPES_H_

/**
 * @file oskar_vector_types.h
 */

#ifdef __CUDACC__
/* Include the CUDA vector types header first, if we're compiling with nvcc. */
#   include <vector_types.h>
#endif

/* Memory alignment macros mirroring those used by CUDA. */
#if !(defined(__VECTOR_TYPES_H__) || defined(__CUDACC__))
#   if defined(__GNUC__)
#       define __align__(n) __attribute__((aligned(n)))
#   elif defined(_MSC_VER)
#       define __align__(n) __declspec(align(n))
#   endif
#   if defined(__GNUC__) || defined(_WIN64)
#       define __builtin_align__(a) __align__(a)
#   else
#       define __builtin_align__(a)
#   endif

/**
 * @brief Two-element structure (single precision).
 *
 * @details
 * Structure used to hold data for a length-2 vector.
 * This must be compatible with the CUDA float2 type.
 */
struct __builtin_align__(8) float2 { float x, y; };
typedef struct float2 float2;

/**
 * @brief Two-element structure (double precision).
 *
 * @details
 * Structure used to hold data for a length-2 vector.
 * This must be compatible with the CUDA double2 type.
 */
struct __builtin_align__(16) double2 { double x, y; };
typedef struct double2 double2;
#endif

/**
 * @brief Four-element complex structure (single precision).
 *
 * @details
 * Structure used to hold data for a length-4 single precision complex vector.
 * When used as a matrix, the elements should be interpreted as:
 *
 *   ( a  b )
 *   ( c  d )
 */
struct __align__(32) float4c { float2 a, b, c, d; };
typedef struct float4c float4c;

/**
 * @brief Four-element complex structure (double precision).
 *
 * @details
 * Structure used to hold data for a length-4 double precision complex vector.
 * When used as a matrix, the elements should be interpreted as:
 *
 *   ( a  b )
 *   ( c  d )
 */
struct __align__(64) double4c { double2 a, b, c, d; };
typedef struct double4c double4c;

#endif /* include guard */
