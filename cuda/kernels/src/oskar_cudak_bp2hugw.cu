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

#include "cuda/kernels/oskar_cudak_bp2hugw.h"
#include "math/core/phase.h"

// Shared memory pointer used by the kernel.
extern __shared__ float2 smem[];

#define PI2 1.570796327f

__global__
void oskar_cudak_bp2hugw(const int na, const float* ax, const float* ay,
        const float* aw, const float* ag, const float2* weights, const int ns,
        const float* saz, const float* sel, const float k,
        const int maxAntennasPerBlock, float2* image)
{
    // Get the pixel (source position) ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    // Get the source position.
    // (NB. Cannot exit on index condition, as all threads are needed later).
    float az = 0.0f, el = 0.0f, zd2 = 0.0f, sinAz, cosAz, cosEl;
    if (s < ns) {
        az = saz[s];
        el = sel[s];
    }
    zd2 = __powf(PI2 - el, 2.0f); // Source zenith distance squared.
    cosEl = __cosf(el);
    __sincosf(az, &sinAz, &cosAz);

    // Initialise shared memory caches.
    // Antenna positions are cached as float2 for speed increase.
    float2 cpx = make_float2(0.0f, 0.0f); // Clear pixel value.
    float2* cwt = smem; // Cached antenna weights.
    float2* cap = cwt + maxAntennasPerBlock; // Cached antenna positions.
    float2* cab = cap + maxAntennasPerBlock; // Cached antenna beam parameters.

    // Cache a block of antenna positions and weights into shared memory.
    for (int as = 0; as < na; as += maxAntennasPerBlock) {
        int antennasInBlock = na - as;
        if (antennasInBlock > maxAntennasPerBlock)
            antennasInBlock = maxAntennasPerBlock;

        // There are blockDim.x threads available - need to copy
        // antennasInBlock pieces of data from global memory.
        for (int t = threadIdx.x; t < antennasInBlock; t += blockDim.x) {
            const int g = as + t; // Global antenna index.
            cwt[t] = weights[g];
            cap[t].x = ax[g];
            cap[t].y = ay[g];
            cab[t].x = aw[g];
            cab[t].y = ag[g];
        }

        // Must synchronise before computing the signal for these antennas.
        __syncthreads();

        // Loop over antennas in block.
        for (int a = 0; a < antennasInBlock; ++a) {
            // Calculate the geometric phase from the source.
            float2 signal, w = cwt[a];
            float arg = GEOMETRIC_PHASE_2D_HORIZONTAL(cap[a].x,
                    cap[a].y, cosEl, sinAz, cosAz, k);
            __sincosf(arg, &signal.y, &signal.x);

            // Calculate the antenna gain in the direction of the source.
            arg = 0.0f; // Prevent underflows.
            if (zd2 * cab[a].x < 30.0f)
                arg = cab[a].y * __expf(-zd2 * cab[a].x);
            signal.x *= arg;
            signal.y *= arg;

            // Perform complex multiply-accumulate.
            cpx.x += (signal.x * w.x - signal.y * w.y);
            cpx.y += (signal.y * w.x + signal.x * w.y);
        }

        // Must synchronise again before loading in a new block of antennas.
        __syncthreads();
    }

    // Copy result into global memory.
    if (s < ns)
        image[s] = cpx;
}
