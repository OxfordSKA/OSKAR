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

#include "cuda/_weights2dHorizontalGeometric.h"
#include "math/core/phase.h"

/**
 * @details
 * This CUDA kernel produces the complex antenna beamforming weights for the
 * given directions, and stores them in device memory.
 * Each thread generates the complex weights for a single antenna and a single
 * beam direction.
 *
 * The number of floating-point operations performed by this kernel is:
 * \li Sines and cosines: 2 * na * nb.
 * \li Multiplies: 4 * na * nb.
 * \li Divides: 2 * na * nb.
 * \li Additions / subtractions: na * nb.
 *
 * @param[in] na Number of antennas.
 * @param[in] ax Array of antenna x positions in metres.
 * @param[in] ay Array of antenna y positions in metres.
 * @param[in] nb Number of beams.
 * @param[in] cbe Cosine of all beam elevations.
 * @param[in] cba Cosine of all beam azimuths.
 * @param[in] sba Sine of all beam azimuths.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] weights Matrix of complex antenna weights (na columns, nb rows).
 */
__global__
void _weights2dHorizontalGeometric(const int na, const float* ax, const float* ay,
        const int nb, const float* cbe, const float* cba, const float* sba,
        const float k, float2* weights)
{
    // Get the antenna and beam ID that this thread is working on.
    const int i = blockDim.x * blockIdx.x + threadIdx.x; // Thread index.
    const int a = i % na; // Antenna index.
    const int b = i / na; // Beam index.
    if (a >= na || b >= nb) return; // Return if either index is out of range.

    // Compute the geometric phase of the beam direction.
    const float phase = -GEOMETRIC_PHASE_2D_HORIZONTAL(ax[a], ay[a],
            cbe[b], sba[b], cba[b], k);
    const int w = a + b*na;
    weights[w].x = cosf(phase) / na; // Normalised real part.
    weights[w].y = sinf(phase) / na; // Normalised imaginary part.
}

/**
 * @details
 * This CUDA kernel produces the complex antenna beamforming weights for the
 * given directions, and stores them in device memory.
 * Each thread generates the complex weights for a single antenna and a single
 * beam direction.
 *
 * The input \p trig array contains triplets of the following pre-computed
 * trigonometry:
 *
 * trig.x = {cosine azimuth}
 * trig.y = {sine azimuth}
 * trig.z = {cosine elevation}
 *
 * The number of floating-point operations performed by this kernel is:
 * \li Sines and cosines: 2 * na * nb.
 * \li Multiplies: 4 * na * nb.
 * \li Divides: 2 * na * nb.
 * \li Additions / subtractions: na * nb.
 *
 * @param[in] na Number of antennas.
 * @param[in] ax Array of antenna x positions in metres.
 * @param[in] ay Array of antenna y positions in metres.
 * @param[in] nb Number of beams.
 * @param[in] trig Precomputed trigonometry for all beam positions.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] weights Matrix of complex antenna weights (na columns, nb rows).
 */
__global__
void _weights2dHorizontalGeometric(const int na, const float* ax, const float* ay,
        const int nb, const float3* trig, const float k, float2* weights)
{
    // Get the antenna and beam ID that this thread is working on.
    const int i = blockDim.x * blockIdx.x + threadIdx.x; // Thread index.
    const int a = i % na; // Antenna index.
    const int b = i / na; // Beam index.
    if (a >= na || b >= nb) return; // Return if either index is out of range.

    // Determine which antennas and beams this thread block is doing.
    const int blockStart = blockDim.x * blockIdx.x;
    const int blockEnd = blockStart + blockDim.x - 1;
    const int antStart = blockStart % na;
    const int beamStart = blockStart / na;
    const int nBeams = blockEnd / na - beamStart + 1;

    __shared__ float3 beamCache[20];
    for (int bi = threadIdx.x; bi < nBeams; bi += blockDim.x) {
        beamCache[bi] = trig[beamStart + bi];
    }

    __syncthreads();

    // Compute the geometric phase of the beam direction.
    const int bi = b - beamStart;
    float sinPhase, cosPhase;
    const float phase = -GEOMETRIC_PHASE_2D_HORIZONTAL(ax[a], ay[a],
            beamCache[bi].z, beamCache[bi].y, beamCache[bi].x, k);
    sincosf(phase, &sinPhase, &cosPhase);

    __shared__ float2 weightsCache[384];
    weightsCache[threadIdx.x].x = cosPhase / na; // Normalised real part.
    weightsCache[threadIdx.x].y = sinPhase / na; // Normalised imaginary part.

    const int w = threadIdx.x + antStart + beamStart*na;
    weights[w] = weightsCache[threadIdx.x];
}
