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

#include "cuda/kernels/oskar_cudak_dftw_2d_seq_in.h"

// Single precision.

// Shared memory pointer used by the kernel.
extern __shared__ float smem[];

__global__
void oskar_cudakf_dftw_2d_seq_in(const int n_in, const float* x_in,
        const float* y_in, const int n_out, const float* x_out,
        const float* y_out, float2* weights)
{
    const int a = blockDim.x * blockIdx.x + threadIdx.x; // Input index.
    const int b = blockDim.y * blockIdx.y + threadIdx.y; // Output index.

    // Cache input and output data from global memory,
    // avoiding shared memory bank conflicts.
    float* cxi = smem;
    float* cyi = &cxi[blockDim.x];
    float* cyo = &cyi[blockDim.x];
    float* cxo = &cyo[blockDim.y];
    if (a < n_in && threadIdx.y == 0)
    {
        cxi[threadIdx.x] = x_in[a];
        cyi[threadIdx.x] = y_in[a];
    }
    if (b < n_out && threadIdx.x == 0)
    {
        cxo[threadIdx.y] = x_out[b];
        cyo[threadIdx.y] = y_out[b];
    }
    __syncthreads();

    // Compute the geometric phase of the output direction.
    float phase;
    phase =  cxi[threadIdx.x] * cxo[threadIdx.y];
    phase += cyi[threadIdx.x] * cyo[threadIdx.y];
    float2 weight;
    sincosf(phase, &weight.y, &weight.x);

    // Write result to global memory.
    if (a < n_in && b < n_out)
    {
        const int w = a + n_in * b;
        weights[w] = weight;
    }
}

// Double precision.

// Shared memory pointer used by the kernel.
extern __shared__ double smemd[];

__global__
void oskar_cudakd_dftw_2d_seq_in(const int n_in, const double* x_in,
        const double* y_in, const int n_out, const double* x_out,
        const double* y_out, double2* weights)
{
    const int a = blockDim.x * blockIdx.x + threadIdx.x; // Input index.
    const int b = blockDim.y * blockIdx.y + threadIdx.y; // Output index.

    // Cache input and output data from global memory,
    // avoiding shared memory bank conflicts.
    double* cxi = smemd;
    double* cyi = &cxi[blockDim.x];
    double* cyo = &cyi[blockDim.x];
    double* cxo = &cyo[blockDim.y];
    if (a < n_in && threadIdx.y == 0)
    {
        cxi[threadIdx.x] = x_in[a];
        cyi[threadIdx.x] = y_in[a];
    }
    if (b < n_out && threadIdx.x == 0)
    {
        cxo[threadIdx.y] = x_out[b];
        cyo[threadIdx.y] = y_out[b];
    }
    __syncthreads();

    // Compute the geometric phase of the output direction.
    double2 weight;
    double phase;
    phase =  cxi[threadIdx.x] * cxo[threadIdx.y];
    phase += cyi[threadIdx.x] * cyo[threadIdx.y];
    sincos(phase, &weight.y, &weight.x);

    // Write result to global memory.
    if (a < n_in && b < n_out)
    {
        const int w = a + n_in * b;
        weights[w] = weight;
    }
}
