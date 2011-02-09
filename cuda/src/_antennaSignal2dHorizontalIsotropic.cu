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

#include "cuda/_antennaSignal2dHorizontalIsotropic.h"
#include "math/core/phase.h"

// Shared memory pointer used by the kernel.
extern __shared__ float2 smem[];

/**
 * @details
 * This CUDA kernel evaluates the antenna signals for the given source and
 * antenna positions. It requires (8 * number_of_threads_per_block) bytes
 * of shared memory to be preallocated by the caller.
 *
 * Each thread evaluates the signal for a single antenna, looping over
 * all the sources.
 *
 * The cosine and sine of the source azimuths, and the cosine
 * of the elevations, must be given as triplets in the \p strig array:
 *
 * strig.x = {cosine azimuth}
 * strig.y = {sine azimuth}
 * strig.z = {cosine elevation}
 *
 * The computed antenna signals are returned in the \p signals array, which
 * must be pre-sized to length 2*na. The values in the \p signals array
 * are alternate (real, imag) pairs for each antenna.
 *
 * The number of floating-point operations performed by this kernel is:
 * \li Sines and cosines:
 * \li Multiplies:
 * \li Additions / subtractions:
 *
 * @param[in] na Number of antennas.
 * @param[in] ax Array of antenna x positions in metres.
 * @param[in] ay Array of antenna y positions in metres.
 * @param[in] ns The number of source positions.
 * @param[in] samp The source amplitudes.
 * @param[in] strig The cosine and sine of the source coordinates.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] signals The computed antenna signals (see note, above).
 */
__global__
void _antennaSignal2dHorizontalIsotropic(const int na, const float* ax,
        const float* ay, const int ns, const float* samp, const float3* strig,
        const float k, float2* signals)
{
    // Get the antenna ID that this thread is working on.
    const int a = blockDim.x * blockIdx.x + threadIdx.x;
    if (a >= na) return; // Return if the index is out of range.

    // Get the antenna position.
    const float x = ax[a];
    const float y = ay[a];

    // Initialise shared memory to hold complex antenna signal.
    smem[threadIdx.x] = make_float2(0.0, 0.0);

    // Loop over all sources.
    float sinPhase, cosPhase;
    for (int s = 0; s < ns; ++s) {
        // Calculate the geometric phase from the source.
        const float phase = GEOMETRIC_PHASE_2D_HORIZONTAL(x, y,
                strig[s].z, strig[s].y, strig[s].x, k);

        // Perform complex multiply-accumulate.
        sincosf(phase, &sinPhase, &cosPhase);
        smem[threadIdx.x].x += (samp[s] * cosPhase);
        smem[threadIdx.x].y += (samp[s] * sinPhase);
    }

    // Copy shared memory back into global memory.
    signals[a].x = smem[threadIdx.x].x;
    signals[a].y = smem[threadIdx.x].y;
}

/**
 * @details
 * Same as above, but using manual caching of source data into shared memory.
 */
__global__
void _antennaSignal2dHorizontalIsotropicCached(const unsigned na,
        const float* ax, const float* ay, const unsigned ns, const float* samp,
        const float3* strig, const float k, const unsigned maxSourcesPerBlock,
        float2* signals)
{
    // Get the antenna ID that this thread is working on.
    const unsigned a = blockDim.x * blockIdx.x + threadIdx.x;

    // Get the antenna position.
    // (NB. Cannot exit on index condition, as all threads are needed later).
    float x = 0.0f, y = 0.0f;
    if (a < na) {
        x = ax[a];
        y = ay[a];
    }

    // Initialise shared memory to hold complex antenna signal.
    float2* csig = smem;
    float4* csrc = (float4*) (&smem[blockDim.x]);
    csig[threadIdx.x] = make_float2(0.0f, 0.0f);

    // Divide source list up into blocks, and cache the contents of each block
    // in shared memory before using it to accumulate the antenna signal.
    unsigned blocks = (ns + maxSourcesPerBlock - 1) / maxSourcesPerBlock;
    for (unsigned block = 0; block < blocks; ++block) {
        const unsigned sourceStart = block * maxSourcesPerBlock;
        unsigned sourcesInBlock = ns - sourceStart;
        if (sourcesInBlock > maxSourcesPerBlock) {
            sourcesInBlock = maxSourcesPerBlock;
        }

        // There are blockDim.x threads available - need to copy
        // sourcesInBlock pieces of data from global memory.
        for (unsigned t = threadIdx.x; t < sourcesInBlock; t += blockDim.x) {
            const unsigned sg = t + sourceStart; // global source index
            csrc[t].x = strig[sg].x;
            csrc[t].y = strig[sg].y;
            csrc[t].z = strig[sg].z;
            csrc[t].w = samp[sg];
        }

        // Must synchronise before computing the signal from these sources.
        __syncthreads();

        // Loop over sources in block.
        for (unsigned s = 0; s < sourcesInBlock; ++s) {
            // Calculate the geometric phase from the source.
            const float phase = GEOMETRIC_PHASE_2D_HORIZONTAL(x, y,
                    csrc[s].z, csrc[s].y, csrc[s].x, k);

            // Perform complex multiply-accumulate.
            float sinPhase, cosPhase;
            __sincosf(phase, &sinPhase, &cosPhase);
            csig[threadIdx.x].x += (csrc[s].w * cosPhase);
            csig[threadIdx.x].y += (csrc[s].w * sinPhase);
        }

        // Must synchronise again before loading in a new block of sources.
        __syncthreads();
    }

    // Copy shared memory back into global memory.
    if (a < na)
        signals[a] = csig[threadIdx.x];
}
