/*
 * Copyright (c) 2012, The University of Oxford
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

#include "station/oskar_evaluate_element_weights_errors_cuda.h"

#include <curand_kernel.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_evaluate_element_weights_errors_cuda_f(float2* errors,
        int num_elements, const float* amp_gain, const float* amp_error,
        const float* phase_offset, const float* phase_error,
        struct curandStateXORWOW* state)
{
    int num_blocks, num_threads = 128; /* Note: this might not be optimal! */
    num_blocks = (num_elements + num_threads - 1) / num_threads;
    oskar_evaluate_element_weights_errors_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (errors, num_elements,
            amp_gain, amp_error, phase_offset, phase_error, state);
}

/* Double precision. */
void oskar_evaluate_element_weights_errors_cuda_d(double2* errors,
        int num_elements, const double* amp_gain, const double* amp_error,
        const double* phase_offset, const double* phase_error,
        struct curandStateXORWOW* state)
{
    int num_blocks, num_threads = 128; /* Note: this might not be optimal! */
    num_blocks = (num_elements + num_threads - 1) / num_threads;
    oskar_evaluate_element_weights_errors_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (errors, num_elements,
            amp_gain, amp_error, phase_offset, phase_error, state);
}


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_evaluate_element_weights_errors_cudak_f(float2* errors,
        int num_elements, const float* amp_gain, const float* amp_error,
        const float* phase_offset, const float* phase_error,
        struct curandStateXORWOW* state)
{
    float amp, arg;
    float2 r;

    /* Thread index == antenna element */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    /* Generate 2 pseudo-random numbers with mean 0.0 and stddev 1.0 */
    r = curand_normal2(&state[idx]);

    /* Evaluate the real and imag. components of the error weight for the antenna */
    amp = amp_gain[idx] + r.x * amp_error[idx];
    arg = phase_offset[idx] + r.y * phase_error[idx];
    errors[idx].x = amp * cosf(arg);
    errors[idx].y = amp * sinf(arg);
}

/* Double precision. */
__global__
void oskar_evaluate_element_weights_errors_cudak_d(double2* errors,
        int num_elements, const double* amp_gain, const double* amp_error,
        const double* phase_offset, const double* phase_error,
        struct curandStateXORWOW* state)
{
    double amp, arg;
    double2 r;

    /* Thread index == antenna element */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_elements) return;

    /* Generate 2 pseudo-random numbers with mean 0.0 and stddev 1.0 */
    r = curand_normal2_double(&state[idx]);

    /* Evaluate the real and imag. components of the error weight for the antenna */
    amp = amp_gain[idx] + (r.x * amp_error[idx]);
    arg = phase_offset[idx] + (r.y * phase_error[idx]);
    errors[idx].x = amp * cos(arg);
    errors[idx].y = amp * sin(arg);
}

#ifdef __cplusplus
}
#endif
