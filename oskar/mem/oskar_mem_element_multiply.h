/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#ifndef OSKAR_MEM_ELEMENT_MULTIPLY_H_
#define OSKAR_MEM_ELEMENT_MULTIPLY_H_

/**
 * @file oskar_mem_element_multiply.h
 */

#include <oskar_global.h>
#include <oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Multiplies (element-wise) the contents of two arrays.
 *
 * @details
 * This function multiplies each element of one array by each element in
 * another array.
 *
 * Using Matlab syntax, this can be expressed as c = a .* b
 *
 * The arrays can be in either CPU or GPU memory, but will be copied to the GPU
 * if necessary before performing the multiplication.
 *
 * If c is NULL on input then the operation becomes a = a .* b.
 *
 * @param[out]    c   Output array.
 * @param[in,out] a   Input and/or output array.
 * @param[in]     b   Second input array.
 * @param[in]     num If >0, use only this number of elements from A and B.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
void oskar_mem_element_multiply(oskar_Mem* c, oskar_Mem* a, const oskar_Mem* b,
        size_t num, int* status);


OSKAR_EXPORT
void oskar_mem_element_multiply_rr_r_f(size_t num, float* c,
        const float* a, const float* b);

OSKAR_EXPORT
void oskar_mem_element_multiply_cc_c_f(size_t num, float2* c,
        const float2* a, const float2* b);

OSKAR_EXPORT
void oskar_mem_element_multiply_cc_m_f(size_t num, float4c* c,
        const float2* a, const float2* b);

OSKAR_EXPORT
void oskar_mem_element_multiply_cm_m_f(size_t num, float4c* c,
        const float2* a, const float4c* b);

OSKAR_EXPORT
void oskar_mem_element_multiply_mm_m_f(size_t num, float4c* c,
        const float4c* a, const float4c* b);


OSKAR_EXPORT
void oskar_mem_element_multiply_rr_r_d(size_t num, double* c,
        const double* a, const double* b);

OSKAR_EXPORT
void oskar_mem_element_multiply_cc_c_d(size_t num, double2* c,
        const double2* a, const double2* b);

OSKAR_EXPORT
void oskar_mem_element_multiply_cc_m_d(size_t num, double4c* c,
        const double2* a, const double2* b);

OSKAR_EXPORT
void oskar_mem_element_multiply_cm_m_d(size_t num, double4c* c,
        const double2* a, const double4c* b);

OSKAR_EXPORT
void oskar_mem_element_multiply_mm_m_d(size_t num, double4c* c,
        const double4c* a, const double4c* b);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_ELEMENT_MULTIPLY_H_ */
