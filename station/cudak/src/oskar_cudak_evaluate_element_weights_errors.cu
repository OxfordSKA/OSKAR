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

#include "station/cudak/oskar_cudak_evaluate_element_weights_errors.h"

#include <curand_kernel.h>

__global__
void oskar_cudak_evaluate_element_weights_errors_d(double2* errors, int n,
        const double* amp_gain, const double* amp_error,
        const double* phase_offset, const double* phase_error,
        curandState* state)
{
    /* Thread index == antenna element */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Return if index is out of range */
    if (idx >= n) return;

    /* Generate 2 pseudo-random numbers with mean 0.0 and stddev 1.0 */
    double2 r = curand_normal2_double(&state[idx]);

    /* Evaluate the real and imag. components of the error weight for the antenna */
    double amp = amp_gain[idx] + (r.x * amp_error[idx]);
    double arg = phase_offset[idx] + (r.y * phase_error[idx]);
    errors[idx].x = amp * cos(arg);
    errors[idx].y = amp * sin(arg);
}

__global__
void oskar_cudak_evaluate_element_weights_errors_f(float2* errors, int n,
        const float* amp_gain, const float* amp_error,
        const float* phase_offset, const float* phase_error,
        curandState* state)
{
    /* Thread index == antenna element */
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    /* Return if index is out of range */
    if (idx >= n) return;

    /* Generate 2 pseudo-random numbers with mean 0.0 and stddev 1.0 */
    float2 r = curand_normal2(&state[idx]);

    /* Evaluate the real and imag. components of the error weight for the antenna */
    float amp = amp_gain[idx] + r.x * amp_error[idx];
    float arg = phase_offset[idx] + r.y * phase_error[idx];
    errors[idx].x = amp * cosf(arg);
    errors[idx].y = amp * sinf(arg);
}

