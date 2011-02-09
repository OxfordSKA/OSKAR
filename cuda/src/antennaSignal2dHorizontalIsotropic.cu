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

#include "cuda/antennaSignal2dHorizontalIsotropic.h"
#include "cuda/_antennaSignal2dHorizontalIsotropic.h"
#include "cuda/_precompute2dHorizontalTrig.h"
#include <cstdio>
#include <vector>

#define TIMER_ENABLE 1
#include "utility/timer.h"

/**
 * @details
 * Computes antenna signals using CUDA.
 *
 * The function must be supplied with the antenna x- and y-positions, the
 * source amplitudes, longitude and latitude positions, and the wavenumber.
 *
 * The computed antenna signals are returned in the \p signals array, which
 * must be pre-sized to length 2*na. The values in the \p signals array
 * are alternate (real, imag) pairs for each antenna.
 *
 * @param[in] na The number of antennas.
 * @param[in] ax The antenna x-positions in metres.
 * @param[in] ay The antenna y-positions in metres.
 * @param[in] ns The number of source positions.
 * @param[in] samp The source amplitudes.
 * @param[in] slon The source longitude coordinates in radians.
 * @param[in] slat The source latitude coordinates in radians.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] signals The computed antenna signals (see note, above).
 */
void antennaSignal2dHorizontalIsotropic(const unsigned na, const float* ax,
        const float* ay, const unsigned ns, const float* samp,
        const float* slon, const float* slat, const float k, float* signals)
{
    // Create source position pairs in host memory.
    std::vector<float2> spos(ns);
    for (unsigned i = 0; i < ns; ++i) spos[i] = make_float2(slon[i], slat[i]);

    // Allocate memory for antenna positions, source positions
    // and antenna signals on the device.
    float *axd, *ayd, *sampd;
    float2 *sig, *sposd;
    float3 *strigd;
    cudaMalloc((void**)&axd, na * sizeof(float));
    cudaMalloc((void**)&ayd, na * sizeof(float));
    cudaMalloc((void**)&sampd, ns * sizeof(float));
    cudaMalloc((void**)&sig, na * sizeof(float2));
    cudaMalloc((void**)&sposd, ns * sizeof(float2));
    cudaMalloc((void**)&strigd, ns * sizeof(float3));

    // Copy antenna positions and source positions to device.
    cudaMemcpy(axd, ax, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ayd, ay, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sposd, &spos[0], ns * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(sampd, samp, ns * sizeof(float), cudaMemcpyHostToDevice);

    // Error code.
    cudaError_t err;

    // Invoke kernel to precompute source positions on the device.
    unsigned sThreadsPerBlock = 384;
    unsigned sBlocks = (ns + sThreadsPerBlock - 1) / sThreadsPerBlock;
    _precompute2dHorizontalTrig <<<sBlocks, sThreadsPerBlock>>>
            (ns, sposd, strigd);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Invoke kernel to compute antenna signals on the device.
    unsigned threadsPerBlock = 384;
    unsigned blocks = (na + threadsPerBlock - 1) / threadsPerBlock;
//    size_t sharedMem = threadsPerBlock * sizeof(float2);
//    _antennaSignal2dHorizontalIsotropic <<<blocks, threadsPerBlock, sharedMem>>>
//            (na, axd, ayd, ns, sampd, strigd, k, sig);
    unsigned maxSourcesPerBlock = 384;
    size_t sharedMem = threadsPerBlock * sizeof(float2)
            + maxSourcesPerBlock * sizeof(float4);
    _antennaSignal2dHorizontalIsotropicCached <<<blocks, threadsPerBlock, sharedMem>>>
            (na, axd, ayd, ns, sampd, strigd, k, maxSourcesPerBlock, sig);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Copy result from device memory to host memory.
    cudaMemcpy(signals, sig, na * sizeof(float2), cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(axd);
    cudaFree(ayd);
    cudaFree(sampd);
    cudaFree(sig);
    cudaFree(sposd);
    cudaFree(strigd);
}
