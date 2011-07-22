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

#include "cuda/kernels/oskar_cudak_as2hi.h"
#include "math/oskar_math_phase.h"

// Shared memory pointer used by the kernel.
extern __shared__ float2 smem[];

__global__
void oskar_cudakf_as2hi(const int na, const float* ax, const float* ay,
        const int ns, const float* samp, const float3* strig, const float k,
        const int maxSourcesPerBlock, float2* signals)
{
    // Get the antenna ID that this thread is working on.
    const int a = blockDim.x * blockIdx.x + threadIdx.x;

    // Get the antenna position.
    // (NB. Cannot exit on index condition, as all threads are needed later).
    float x = 0.0f, y = 0.0f;
    if (a < na) {
        x = ax[a];
        y = ay[a];
    }

    // Initialise shared memory to hold complex antenna signal.
    float2 csig = make_float2(0.0f, 0.0f);
    float4* csrc = (float4*) smem;

    // Divide source list up into blocks, and cache the contents of each block
    // in shared memory before using it to accumulate the antenna signal.
    int blocks = (ns + maxSourcesPerBlock - 1) / maxSourcesPerBlock;
    for (int block = 0; block < blocks; ++block) {
        const int sourceStart = block * maxSourcesPerBlock;
        int sourcesInBlock = ns - sourceStart;
        if (sourcesInBlock > maxSourcesPerBlock)
            sourcesInBlock = maxSourcesPerBlock;

        // There are blockDim.x threads available - need to copy
        // sourcesInBlock pieces of data from global memory.
        for (int t = threadIdx.x; t < sourcesInBlock; t += blockDim.x) {
            const int sg = t + sourceStart; // global source index
            csrc[t].x = strig[sg].x;
            csrc[t].y = strig[sg].y;
            csrc[t].z = strig[sg].z;
            csrc[t].w = samp[sg];
        }

        // Must synchronise before computing the signal from these sources.
        __syncthreads();

        // Loop over sources in block.
        for (int s = 0; s < sourcesInBlock; ++s) {
            // Calculate the geometric phase from the source.
            const float phase = GEOMETRIC_PHASE_2D_HORIZONTAL(x, y,
                    csrc[s].z, csrc[s].y, csrc[s].x, k);

            // Perform complex multiply-accumulate.
            float sinPhase, cosPhase;
            sincosf(phase, &sinPhase, &cosPhase);
            csig.x += (csrc[s].w * cosPhase);
            csig.y += (csrc[s].w * sinPhase);
        }

        // Must synchronise again before loading in a new block of sources.
        __syncthreads();
    }

    // Copy result into global memory.
    if (a < na)
        signals[a] = csig;
}
