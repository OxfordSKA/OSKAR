/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#ifndef OSKAR_EVALUATE_ELEMENT_WEIGHTS_ERRORS_CUDA_H_
#define OSKAR_EVALUATE_ELEMENT_WEIGHTS_ERRORS_CUDA_H_

/**
 * @file oskar_evaluate_element_weights_errors_cuda.h
 */

#include <oskar_global.h>
#include <utility/oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

OSKAR_EXPORT
void oskar_evaluate_element_weights_errors_cuda_f(int num_elements,
        const float* amp_gain, const float* amp_error,
        const float* phase_offset, const float* phase_error, float2* errors);

OSKAR_EXPORT
void oskar_evaluate_element_weights_errors_cuda_d(int num_elements,
        const double* amp_gain, const double* amp_error,
        const double* phase_offset, const double* phase_error, double2* errors);

#ifdef __CUDACC__

__global__
void oskar_evaluate_element_weights_errors_cudak_f(int num_elements,
        const float* restrict amp_gain, const float* restrict amp_error,
        const float* restrict phase_offset, const float* restrict phase_error,
        float2* errors);

__global__
void oskar_evaluate_element_weights_errors_cudak_d(int num_elements,
        const double* restrict amp_gain, const double* restrict amp_error,
        const double* restrict phase_offset, const double* restrict phase_error,
        double2* errors);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_ELEMENT_WEIGHTS_ERRORS_CUDA_H_ */
