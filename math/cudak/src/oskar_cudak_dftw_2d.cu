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

#include "math/cudak/oskar_cudak_dftw_2d_seq_in.h"

// Single precision.
__global__
void oskar_cudak_dftw_2d_f(const int n_in, const float* x_in,
        const float* y_in, const float x_out, const float y_out,
        float2* weights)
{
    // Get input index.
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_in) return;

    // Cache input data from global memory.
    float cxi = x_in[i];
    float cyi = y_in[i];

    // Compute the geometric phase of the output direction.
    float phase;
    phase =  cxi * x_out;
    phase += cyi * y_out;
    float2 weight;
    sincosf(phase, &weight.y, &weight.x);

    // Write result to global memory.
    weights[i] = weight;
}

// Double precision.
__global__
void oskar_cudak_dftw_2d_d(const int n_in, const double* x_in,
        const double* y_in, const double x_out, const double y_out,
        double2* weights)
{
    // Get input index.
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n_in) return;

    // Cache input data from global memory.
    double cxi = x_in[i];
    double cyi = y_in[i];

    // Compute the geometric phase of the output direction.
    double phase;
    phase =  cxi * x_out;
    phase += cyi * y_out;
    double2 weight;
    sincos(phase, &weight.y, &weight.x);

    // Write result to global memory.
    weights[i] = weight;
}
