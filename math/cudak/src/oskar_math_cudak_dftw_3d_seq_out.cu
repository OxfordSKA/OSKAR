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

#include "cuda/kernels/oskar_cudak_dftw_3d_seq_out.h"

// Single precision.

// Shared memory pointer used by the kernel.
extern __shared__ float smem[];

__global__
void oskar_cudakf_dftw_3d_seq_out(const int n_in, const float* x_in,
        const float* y_in, const float* z_in, const int n_out,
        const float* x_out, const float* y_out, const float* z_out,
        float2* weights)
{
    const int s = blockDim.x * blockIdx.x + threadIdx.x; // Output index.
    const int a = blockDim.y * blockIdx.y + threadIdx.y; // Input index.

    // Cache input and output data from global memory.
    float* cxo = smem;
    float* cyo = &cxo[blockDim.x];
    float* czo = &cyo[blockDim.x];
    float* cxi = &czo[blockDim.x];
    float* cyi = &cxi[blockDim.y];
    float* czi = &cyi[blockDim.y];
    if (s < n_out && threadIdx.y == 0)
    {
        cxo[threadIdx.x] = x_out[s];
        cyo[threadIdx.x] = y_out[s];
        czo[threadIdx.x] = z_out[s];
    }
    if (a < n_in && threadIdx.x == 0)
    {
        cxi[threadIdx.y] = x_in[a];
        cyi[threadIdx.y] = y_in[a];
        czi[threadIdx.y] = z_in[a];
    }
    __syncthreads();

    // Compute the geometric phase of the output direction.
    float phase;
    phase =  cxi[threadIdx.y] * cxo[threadIdx.x];
    phase += cyi[threadIdx.y] * cyo[threadIdx.x];
    phase += czi[threadIdx.y] * czo[threadIdx.x];
    float2 weight;
    sincosf(phase, &weight.y, &weight.x);

    // Write result to global memory.
    if (s < n_out && a < n_in)
    {
        const int w = s + n_out * a;
        weights[w] = weight;
    }
}

// Double precision.

// Shared memory pointer used by the kernel.
extern __shared__ double smemd[];

__global__
void oskar_cudakd_dftw_3d_seq_out(const int n_in, const double* x_in,
        const double* y_in, const double* z_in, const int n_out,
        const double* x_out, const double* y_out, const double* z_out,
        double2* weights)
{
    const int s = blockDim.x * blockIdx.x + threadIdx.x; // Output index.
    const int a = blockDim.y * blockIdx.y + threadIdx.y; // Input index.

    // Cache input and output data from global memory.
    double* cxo = smemd;
    double* cyo = &cxo[blockDim.x];
    double* czo = &cyo[blockDim.x];
    double* cxi = &czo[blockDim.x];
    double* cyi = &cxi[blockDim.y];
    double* czi = &cyi[blockDim.y];
    if (s < n_out && threadIdx.y == 0)
    {
        cxo[threadIdx.x] = x_out[s];
        cyo[threadIdx.x] = y_out[s];
        czo[threadIdx.x] = z_out[s];
    }
    if (a < n_in && threadIdx.x == 0)
    {
        cxi[threadIdx.y] = x_in[a];
        cyi[threadIdx.y] = y_in[a];
        czi[threadIdx.y] = z_in[a];
    }
    __syncthreads();

    // Compute the geometric phase of the output direction.
    double phase;
    phase =  cxi[threadIdx.y] * cxo[threadIdx.x];
    phase += cyi[threadIdx.y] * cyo[threadIdx.x];
    phase += czi[threadIdx.y] * czo[threadIdx.x];
    double2 weight;
    sincos(phase, &weight.y, &weight.x);

    // Write result to global memory.
    if (s < n_out && a < n_in)
    {
        const int w = s + n_out * a;
        weights[w] = weight;
    }
}
