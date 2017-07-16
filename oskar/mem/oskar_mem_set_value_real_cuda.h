/*
 * Copyright (c) 2013, The University of Oxford
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

#ifndef OSKAR_MEM_SET_VALUE_REAL_CUDA_H_
#define OSKAR_MEM_SET_VALUE_REAL_CUDA_H_

/**
 * @file oskar_mem_set_value_real_cuda.h
 */

#include <oskar_global.h>
#include <utility/oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Single precision. */
OSKAR_EXPORT
void oskar_mem_set_value_real_cuda_r_f(int num, float* data, float value);

OSKAR_EXPORT
void oskar_mem_set_value_real_cuda_c_f(int num, float2* data, float value);

OSKAR_EXPORT
void oskar_mem_set_value_real_cuda_m_f(int num, float4c* data, float value);


/* Double precision. */
OSKAR_EXPORT
void oskar_mem_set_value_real_cuda_r_d(int num, double* data, double value);

OSKAR_EXPORT
void oskar_mem_set_value_real_cuda_c_d(int num, double2* data, double value);

OSKAR_EXPORT
void oskar_mem_set_value_real_cuda_m_d(int num, double4c* data, double value);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MEM_SET_VALUE_REAL_CUDA_H_ */
