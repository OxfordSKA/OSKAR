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

#include "cuda/kernels/oskar_cudak_wt2phg.h"
#include "math/oskar_phase.h"

// Single precision.

// Shared memory pointer used by the kernel.
extern __shared__ float smem[];

__global__
void oskar_cudakf_wt2phg(const int na, const float* ax, const float* ay,
        const int nb, const float* bcace, const float* bsace, float2* weights)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int a = blockDim.x * blockIdx.x + tx; // Antenna index.
    const int b = blockDim.y * blockIdx.y + ty; // Beam index.

    // Cache antenna and beam data from global memory,
    // avoiding shared memory bank conflicts.
    float* cax = smem;
    float* cay = &cax[blockDim.x];
    float* ccace = &cay[blockDim.x];
    float* csace = &ccace[blockDim.y];
    if (a < na) {
        cax[tx] = ax[a];
        cay[tx] = ay[a];
    }
    if (b < nb) {
        csace[ty] = bsace[b];
        ccace[ty] = bcace[b];
    }
    __syncthreads();

    // Compute the geometric phase of the beam direction.
    float2 weight;
    const float phase = cax[tx] * csace[ty] + cay[tx] * ccace[ty];
    sincosf(phase, &weight.y, &weight.x);
    // Do NOT normalise.

    // Write result to global memory.
    if (a < na && b < nb) {
        const int w = a + na * b;
        weights[w] = weight;
    }
}

// Double precision.

// Shared memory pointer used by the kernel.
extern __shared__ double smemd[];

__global__
void oskar_cudakd_wt2phg(const int na, const double* ax, const double* ay,
        const int nb, const double* bcace, const double* bsace,
        double2* weights)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int a = blockDim.x * blockIdx.x + tx; // Antenna index.
    const int b = blockDim.y * blockIdx.y + ty; // Beam index.

    // Cache antenna and beam data from global memory,
    // avoiding shared memory bank conflicts.
    double* cax = smemd;
    double* cay = &cax[blockDim.x];
    double* ccace = &cay[blockDim.x];
    double* csace = &ccace[blockDim.y];
    if (a < na) {
        cax[tx] = ax[a];
        cay[tx] = ay[a];
    }
    if (b < nb) {
        csace[ty] = bsace[b];
        ccace[ty] = bcace[b];
    }
    __syncthreads();

    // Compute the geometric phase of the beam direction.
    double2 weight;
    const double phase = cax[tx] * csace[ty] + cay[tx] * ccace[ty];
    sincos(phase, &weight.y, &weight.x);
    // Do NOT normalise.

    // Write result to global memory.
    if (a < na && b < nb) {
        const int w = a + na * b;
        weights[w] = weight;
    }
}
