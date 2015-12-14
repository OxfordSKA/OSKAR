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

#include <oskar_evaluate_element_weights_errors_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_evaluate_element_weights_errors_cuda_f(int num_elements,
        const float* amp_gain, const float* amp_error,
        const float* phase_offset, const float* phase_error, float2* errors)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_elements + num_threads - 1) / num_threads;
    oskar_evaluate_element_weights_errors_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_elements,
            amp_gain, amp_error, phase_offset, phase_error, errors);
}

/* Double precision. */
void oskar_evaluate_element_weights_errors_cuda_d(int num_elements,
        const double* amp_gain, const double* amp_error,
        const double* phase_offset, const double* phase_error, double2* errors)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_elements + num_threads - 1) / num_threads;
    oskar_evaluate_element_weights_errors_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_elements,
            amp_gain, amp_error, phase_offset, phase_error, errors);
}


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_evaluate_element_weights_errors_cudak_f(int num_elements,
        const float* restrict amp_gain, const float* restrict amp_error,
        const float* restrict phase_offset, const float* restrict phase_error,
        float2* errors)
{
    float2 r, t;

    /* Thread index is antenna element. */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    /* Get two random numbers from a normalised Gaussian distribution. */
    r = errors[i];

    /* Evaluate the real and imaginary components of the error weight
     * for the antenna. */
    r.x *= amp_error[i];
    r.x += amp_gain[i]; /* Amplitude. */
    r.y *= phase_error[i];
    r.y += phase_offset[i]; /* Phase. */
    sincosf(r.y, &t.y, &t.x);
    t.x *= r.x; /* Real. */
    t.y *= r.x; /* Imaginary. */
    errors[i] = t; /* Store. */
}

/* Double precision. */
__global__
void oskar_evaluate_element_weights_errors_cudak_d(int num_elements,
        const double* restrict amp_gain, const double* restrict amp_error,
        const double* restrict phase_offset, const double* restrict phase_error,
        double2* errors)
{
    double2 r, t;

    /* Thread index is antenna element. */
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_elements) return;

    /* Get two random numbers from a normalised Gaussian distribution. */
    r = errors[i];

    /* Evaluate the real and imaginary components of the error weight
     * for the antenna. */
    r.x *= amp_error[i];
    r.x += amp_gain[i]; /* Amplitude. */
    r.y *= phase_error[i];
    r.y += phase_offset[i]; /* Phase. */
    sincos(r.y, &t.y, &t.x);
    t.x *= r.x; /* Real. */
    t.y *= r.x; /* Imaginary. */
    errors[i] = t; /* Store. */
}

#ifdef __cplusplus
}
#endif
