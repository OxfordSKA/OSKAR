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

#include "cuda/kernels/oskar_cudak_rpw3leg.h"
#include "math/oskar_math_phase.h"

// Single precision.

// Shared memory pointer used by the kernel.
extern __shared__ float smem[];

__global__
void oskar_cudakf_rpw3leg(const int na, const float* ax, const float* ay,
        const float* az, const float2 scha0, const float2 scdec0, const int ns,
        const float* ha, const float* dec, const float k, float2* weights)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int s = blockDim.x * blockIdx.x + tx; // Source index.
    const int a = blockDim.y * blockIdx.y + ty; // Antenna index.

    // Cache source and antenna data from global memory,
    // avoiding shared memory bank conflicts.
    float* cha = smem;
    float* cdc = &cha[blockDim.x];
    float* cax = &cdc[blockDim.x];
    float* cay = &cax[blockDim.y];
    float* caz = &cay[blockDim.y];
    if (s < ns) {
        cha[tx] = ha[s];
        cdc[tx] = dec[s];
    }
    if (a < na) {
        cax[ty] = ax[a];
        cay[ty] = ay[a];
        caz[ty] = az[a];
    }
    __syncthreads();

    // Compute the geometric phase of the reference direction.
    const float phase0 = GEOMETRIC_PHASE_3D_LOCAL_EQUATORIAL_K(
            cax[ty], cay[ty], caz[ty], scdec0.x, scdec0.y, scha0.x, scha0.y);

    // Compute the geometric phase of the source direction.
    float2 sha, sdc, weight;
    sincosf(cha[tx], &sha.x, &sha.y);
    sincosf(cdc[tx], &sdc.x, &sdc.y);
    const float phase = GEOMETRIC_PHASE_3D_LOCAL_EQUATORIAL_K(
            cax[ty], cay[ty], caz[ty], sdc.x, sdc.y, sha.x, sha.y);

    // Compute the relative phase.
    const float arg = -k * (phase - phase0);
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
void oskar_cudakd_rpw3leg(const int na, const double* ax, const double* ay,
        const double* az, const double2 scha0, const double2 scdec0, const int ns,
        const double* ha, const double* dec, const double k, double2* weights)
{
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int s = blockDim.x * blockIdx.x + tx; // Source index.
    const int a = blockDim.y * blockIdx.y + ty; // Antenna index.

    // Cache source and antenna data from global memory,
    // avoiding shared memory bank conflicts.
    double* cha = smemd;
    double* cdc = &cha[blockDim.x];
    double* cax = &cdc[blockDim.x];
    double* cay = &cax[blockDim.y];
    double* caz = &cay[blockDim.y];
    if (s < ns) {
        cha[tx] = ha[s];
        cdc[tx] = dec[s];
    }
    if (a < na) {
        cax[ty] = ax[a];
        cay[ty] = ay[a];
        caz[ty] = az[a];
    }
    __syncthreads();

    // Compute the geometric phase of the reference direction.
    const double phase0 = GEOMETRIC_PHASE_3D_LOCAL_EQUATORIAL_K(
            cax[ty], cay[ty], caz[ty], scdec0.x, scdec0.y, scha0.x, scha0.y);

    // Compute the geometric phase of the source direction.
    double2 sha, sdc, weight;
    sincos(cha[tx], &sha.x, &sha.y);
    sincos(cdc[tx], &sdc.x, &sdc.y);
    const double phase = GEOMETRIC_PHASE_3D_LOCAL_EQUATORIAL_K(
            cax[ty], cay[ty], caz[ty], sdc.x, sdc.y, sha.x, sha.y);

    // Compute the relative phase.
    const double arg = -k * (phase - phase0);
    sincos(arg, &weight.y, &weight.x);

    // Write result to global memory.
    if (s < ns && a < na) {
        const int w = s + ns * a;
        weights[w] = weight;
    }
}
