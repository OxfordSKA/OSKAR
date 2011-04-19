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

#include "cuda/kernels/oskar_cudak_rpw3leglm.h"
#include "math/core/phase.h"

// Single precision.

// Shared memory pointer used by the kernel.
extern __shared__ float smem[];

__global__
void oskar_cudakf_rpw3leglm(const int na, const float* uvw, const int ns,
        const float* l, const float* m, const float* n, const float k,
        float2* weights)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int s = blockDim.x * blockIdx.x + tx; // Source index.
    const int a = blockDim.y * blockIdx.y + ty; // Antenna index.

    // Cache source and antenna data from global memory.
    float* cl = smem;
    float* cm = &cl[blockDim.x];
    float* cn = &cm[blockDim.x];
    float* cu = &cn[blockDim.x];
    float* cv = &cu[blockDim.y];
    float* cw = &cv[blockDim.y];
    if (s < ns && ty == 0) {
        cl[tx] = l[s];
        cm[tx] = m[s];
        cn[tx] = n[s];
    }
    if (a < na && tx == 0) {
        cu[ty] = uvw[a];
        cv[ty] = uvw[a + na];
        cw[ty] = uvw[a + 2*na];
    }
    __syncthreads();

    float arg = k * (cu[ty] * cl[tx] + cv[ty] * cm[tx] + cw[ty] * cn[tx]);
    float2 weight;
    sincosf(arg, &weight.y, &weight.x);

    // Write result to global memory.
    if (s < ns && a < na) {
        const int w = s + ns * a;
        weights[w] = weight;
    }
}

// Double precision.

// Shared memory pointer used by the kernel.
extern __shared__ double smemd[];

__global__
void oskar_cudakd_rpw3leglm(const int na, const double* uvw, const int ns,
        const double* l, const double* m, const double* n, const double k,
        double2* weights)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int s = blockDim.x * blockIdx.x + tx; // Source index.
    const int a = blockDim.y * blockIdx.y + ty; // Antenna index.

    // Cache source and antenna data from global memory.
    double* cl = smemd;
    double* cm = &cl[blockDim.x];
    double* cn = &cm[blockDim.x];
    double* cu = &cn[blockDim.x];
    double* cv = &cu[blockDim.y];
    double* cw = &cv[blockDim.y];
    if (s < ns && ty == 0) {
        cl[tx] = l[s];
        cm[tx] = m[s];
        cn[tx] = n[s];
    }
    if (a < na && tx == 0) {
        cu[ty] = uvw[a];
        cv[ty] = uvw[a + na];
        cw[ty] = uvw[a + 2*na];
    }
    __syncthreads();

    double arg = k * (cu[ty] * cl[tx] + cv[ty] * cm[tx] + cw[ty] * cn[tx]);
    double2 weight;
    sincos(arg, &weight.y, &weight.x);

    // Write result to global memory.
    if (s < ns && a < na) {
        const int w = s + ns * a;
        weights[w] = weight;
    }
}
