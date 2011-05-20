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

#include "cuda/kernels/oskar_cudak_bp2phiw.h"
#include "math/core/phase.h"

// Single precision.

// Shared memory pointer used by the kernel.
extern __shared__ float2 smem[];

__global__
void oskar_cudakf_bp2phiw(const int na, const float* ax, const float* ay,
        const float2* weights, const int ns, const float* scace,
        const float* ssace, const int maxAntennasPerBlock, float2* image)
{
    // Get the pixel (source position) ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    // Get the source position.
    // (NB. Cannot exit on index condition, as all threads are needed later).
    float sinAzCosEl = 0.0f, cosAzCosEl = 0.0f;
    if (s < ns)
    {
        cosAzCosEl = scace[s];
        sinAzCosEl = ssace[s];
    }

    // Initialise shared memory caches.
    // Antenna positions are cached as float2 for speed increase.
    float2 cpx = make_float2(0.0f, 0.0f); // Clear pixel value.
    float2* cwt = smem; // Cached antenna weights.
    float2* cap = cwt + maxAntennasPerBlock; // Cached antenna positions.

    // Cache a block of antenna positions and weights into shared memory.
    for (int as = 0; as < na; as += maxAntennasPerBlock)
    {
        int antennasInBlock = na - as;
        if (antennasInBlock > maxAntennasPerBlock)
            antennasInBlock = maxAntennasPerBlock;

        // There are blockDim.x threads available - need to copy
        // antennasInBlock pieces of data from global memory.
        for (int t = threadIdx.x; t < antennasInBlock; t += blockDim.x)
        {
            const int ag = as + t; // Global antenna index.
            cwt[t] = weights[ag];
            cap[t].x = ax[ag];
            cap[t].y = ay[ag];
        }

        // Must synchronise before computing the signal for these antennas.
        __syncthreads();

        // Loop over antennas in block.
        for (int a = 0; a < antennasInBlock; ++a)
        {
            // Calculate the geometric phase from the source.
            float2 signal, w = cwt[a];
            float phaseSrc = -(cap[a].x * sinAzCosEl + cap[a].y * cosAzCosEl);
            sincosf(phaseSrc, &signal.y, &signal.x);

            // Perform complex multiply-accumulate.
            cpx.x += (signal.x * w.x);
            cpx.x -= (signal.y * w.y);
            cpx.y += (signal.y * w.x);
            cpx.y += (signal.x * w.y);
        }

        // Must synchronise again before loading in a new block of antennas.
        __syncthreads();
    }

    // Copy result into global memory.
    if (s < ns)
        image[s] = cpx;
}

// Double precision.

// Shared memory pointer used by the kernel.
extern __shared__ double2 smemd[];

__global__
void oskar_cudakd_bp2phiw(const int na, const double* ax, const double* ay,
        const double2* weights, const int ns, const double* scace,
        const double* ssace, const int maxAntennasPerBlock, double2* image)
{
    // Get the pixel (source position) ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    // Get the source position.
    // (NB. Cannot exit on index condition, as all threads are needed later).
    double sinAzCosEl = 0.0, cosAzCosEl = 0.0;
    if (s < ns)
    {
        cosAzCosEl = scace[s];
        sinAzCosEl = ssace[s];
    }

    // Initialise shared memory caches.
    // Antenna positions are cached as double2 for speed increase.
    double2 cpx = make_double2(0.0, 0.0); // Clear pixel value.
    double2* cwt = smemd; // Cached antenna weights.
    double2* cap = cwt + maxAntennasPerBlock; // Cached antenna positions. (Old)
//    double2* csincos = cwt + maxAntennasPerBlock; // New.

    // Cache a block of antenna positions and weights into shared memory.
    for (int as = 0; as < na; as += maxAntennasPerBlock)
    {
        int antennasInBlock = na - as;
        if (antennasInBlock > maxAntennasPerBlock)
            antennasInBlock = maxAntennasPerBlock;

        // There are blockDim.x threads available - need to copy
        // antennasInBlock pieces of data from global memory.
        for (int t = threadIdx.x; t < antennasInBlock; t += blockDim.x)
        {
            const int ag = as + t; // Global antenna index.
//            double phaseSrc = -(ax[ag] * sinAzCosEl + ay[ag] * cosAzCosEl); // New.
//            sincos(phaseSrc, &csincos[t].y, &csincos[t].x); // New.
            cwt[t] = weights[ag];
            cap[t].x = ax[ag]; // Old.
            cap[t].y = ay[ag]; // Old.
        }

        // Must synchronise before computing the signal for these antennas.
        __syncthreads();

        // Loop over antennas in block.
        for (int a = 0; a < antennasInBlock; ++a)
        {
            // Calculate the geometric phase from the source.
//            double2 signal = csincos[a], w = cwt[a]; // New.
            double2 signal, w = cwt[a];
            double phaseSrc = -(cap[a].x * sinAzCosEl + cap[a].y * cosAzCosEl);
            sincos(phaseSrc, &signal.y, &signal.x);

            // Perform complex multiply-accumulate.
            cpx.x += (signal.x * w.x);
            cpx.y += (signal.y * w.x);
            cpx.x -= (signal.y * w.y);
            cpx.y += (signal.x * w.y);
        }

        // Must synchronise again before loading in a new block of antennas.
        __syncthreads();
    }

    // Copy result into global memory.
    if (s < ns)
        image[s] = cpx;
}
