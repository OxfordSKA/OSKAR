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

#include "cuda/kernels/oskar_cudak_bp2hiws.h"
#include "math/core/phase.h"

// Single precision.

// Shared memory pointer used by the kernel.
extern __shared__ float2 smem[];

__global__
void oskar_cudakf_bp2hiws(const int na, const float* ax, const float* ay,
        const float2* weights, const float2* signals, const int sigStride,
        const int ns, const float* saz, const float* sel, const float k,
        const int maxAntennasPerBlock, float2* image)
{
    // Get the pixel (source position) ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    // Get the source position.
    // (NB. Cannot exit on index condition, as all threads are needed later).
    float az = 0.0f, el = 0.0f, sinAz, cosAz, cosEl;
    if (s < ns) {
        az = saz[s];
        el = sel[s];
    }
    cosEl = cosf(el);
    sincosf(az, &sinAz, &cosAz);

    // Initialise shared memory caches.
    // Antenna positions are cached as float2 for speed increase.
    float2 cpx = make_float2(0.0f, 0.0f); // Clear pixel value.
    float2* cwt = smem; // Cached antenna weights.
    float2* cap = cwt + maxAntennasPerBlock; // Cached antenna positions.

    // Cache a block of antenna positions and weights into shared memory.
    for (int as = 0; as < na; as += maxAntennasPerBlock) {
        int antennasInBlock = na - as;
        if (antennasInBlock > maxAntennasPerBlock)
            antennasInBlock = maxAntennasPerBlock;

        // There are blockDim.x threads available - need to copy
        // antennasInBlock pieces of data from global memory.
        for (int t = threadIdx.x; t < antennasInBlock; t += blockDim.x) {
            const int ag = as + t; // Global antenna index.
            cwt[t] = weights[ag];
            cap[t].x = ax[ag];
            cap[t].y = ay[ag];
        }

        // Must synchronise before computing the signal for these antennas.
        __syncthreads();

        // Loop over antennas in block.
        for (int a = 0; a < antennasInBlock; ++a) {
            float2 w = cwt[a];
            float2 signal = signals[(as + a) * sigStride + s];

            float2 srcSig;
            float phaseSrc = GEOMETRIC_PHASE_2D_HORIZONTAL(cap[a].x,
                    cap[a].y, cosEl, sinAz, cosAz, k);
            sincosf(phaseSrc, &srcSig.y, &srcSig.x);

            float2 signalNew;
            signalNew.x = (signal.x * srcSig.x - signal.y * srcSig.y);
            signalNew.y = (signal.y * srcSig.x + signal.x * srcSig.y);

            // Perform complex multiply-accumulate.
            cpx.x += (signalNew.x * w.x - signalNew.y * w.y);
            cpx.y += (signalNew.y * w.x + signalNew.x * w.y);
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
void oskar_cudakd_bp2hiws(const int na, const double* ax, const double* ay,
        const double2* weights, const double2* signals, const int sigStride,
        const int ns, const double* saz, const double* sel, const double k,
        const int maxAntennasPerBlock, double2* image)
{
    // Get the pixel (source position) ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    // Get the source position.
    // (NB. Cannot exit on index condition, as all threads are needed later).
    double az = 0.0, el = 0.0, sinAz, cosAz, cosEl;
    if (s < ns) {
        az = saz[s];
        el = sel[s];
    }
    cosEl = cos(el);
    sincos(az, &sinAz, &cosAz);

    // Initialise shared memory caches.
    // Antenna positions are cached as double2 for speed increase.
    double2 cpx = make_double2(0.0, 0.0); // Clear pixel value.
    double2* cwt = smemd; // Cached antenna weights.
    double2* cap = cwt + maxAntennasPerBlock; // Cached antenna positions.

    // Cache a block of antenna positions and weights into shared memory.
    for (int as = 0; as < na; as += maxAntennasPerBlock) {
        int antennasInBlock = na - as;
        if (antennasInBlock > maxAntennasPerBlock)
            antennasInBlock = maxAntennasPerBlock;

        // There are blockDim.x threads available - need to copy
        // antennasInBlock pieces of data from global memory.
        for (int t = threadIdx.x; t < antennasInBlock; t += blockDim.x) {
            const int ag = as + t; // Global antenna index.
            cwt[t] = weights[ag];
            cap[t].x = ax[ag];
            cap[t].y = ay[ag];
        }

        // Must synchronise before computing the signal for these antennas.
        __syncthreads();

        // Loop over antennas in block.
        for (int a = 0; a < antennasInBlock; ++a) {
            double2 w = cwt[a];
            double2 signal = signals[(as + a) * sigStride + s];

            double2 srcSig;
            double phaseSrc = GEOMETRIC_PHASE_2D_HORIZONTAL(cap[a].x,
                    cap[a].y, cosEl, sinAz, cosAz, k);
            sincos(phaseSrc, &srcSig.y, &srcSig.x);

            double2 signalNew;
            signalNew.x = (signal.x * srcSig.x - signal.y * srcSig.y);
            signalNew.y = (signal.y * srcSig.x + signal.x * srcSig.y);

            // Perform complex multiply-accumulate.
            cpx.x += (signalNew.x * w.x - signalNew.y * w.y);
            cpx.y += (signalNew.y * w.x + signalNew.x * w.y);
        }

        // Must synchronise again before loading in a new block of antennas.
        __syncthreads();
    }

    // Copy result into global memory.
    if (s < ns)
        image[s] = cpx;
}
