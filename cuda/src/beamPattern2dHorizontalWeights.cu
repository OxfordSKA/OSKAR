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

#include "cuda/beamPattern2dHorizontalWeights.h"
#include "cuda/_beamPattern2dHorizontalWeights.h"
#include "cuda/_weights2dHorizontalGeometric.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Computes a beam pattern using CUDA, generating the beamforming weights
 * separately.
 *
 * The function must be supplied with the antenna x- and y-positions, the
 * test source longitude and latitude positions, the beam direction, and
 * the wavenumber.
 *
 * The computed beam pattern is returned in the \p image array, which
 * must be pre-sized to length 2*ns. The values in the \p image array
 * are alternate (real, imag) pairs for each position of the test source.
 *
 * @param[in] na The number of antennas.
 * @param[in] ax The antenna x-positions in metres.
 * @param[in] ay The antenna y-positions in metres.
 * @param[in] ns The number of test source positions.
 * @param[in] slon The longitude coordinates of the test source.
 * @param[in] slat The latitude coordinates of the test source.
 * @param[in] ba The beam azimuth direction in radians
 * @param[in] be The beam elevation direction in radians.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] image The computed beam pattern (see note, above).
 */
void beamPattern2dHorizontalWeights(const int na, const float* ax,
        const float* ay, const int ns, const float* slon, const float* slat,
        const float ba, const float be, const float k, float* image)
{
    // Precompute.
    float sinBeamAz = sin(ba);
    float cosBeamAz = cos(ba);
    float cosBeamEl = cos(be);

    // Allocate memory for antenna positions, antenna weights,
    // test source positions and pixel values on the device.
    float *axd, *ayd, *slond, *slatd, *sbad, *cbad, *cbed;
    float2 *weights, *pix;
    cudaMalloc((void**)&axd, na * sizeof(float));
    cudaMalloc((void**)&ayd, na * sizeof(float));
    cudaMalloc((void**)&weights, na * sizeof(float2));
    cudaMalloc((void**)&sbad, 1 * sizeof(float));
    cudaMalloc((void**)&cbad, 1 * sizeof(float));
    cudaMalloc((void**)&cbed, 1 * sizeof(float));

    // Copy antenna positions and test source positions to device.
    cudaMemcpy(axd, ax, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ayd, ay, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sbad, &sinBeamAz, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cbad, &cosBeamAz, 1 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cbed, &cosBeamEl, 1 * sizeof(float), cudaMemcpyHostToDevice);

    // Invoke kernel to compute antenna weights on the device.
    int wThreadsPerBlock = 256;
    int wBlocks = (na + wThreadsPerBlock - 1) / wThreadsPerBlock;
    _weights2dHorizontalGeometric <<<wBlocks, wThreadsPerBlock>>> (
            na, axd, ayd, 1, cbed, cbad, sbad, k, weights);
    cudaThreadSynchronize();

    // Divide up the source (pixel) list into manageable chunks.
    int nsMax = 100000;
    int chunk = 0, chunks = (ns + nsMax - 1) / nsMax;

    // Allocate memory for source position chunk on the device.
    cudaMalloc((void**)&slond, nsMax * sizeof(float));
    cudaMalloc((void**)&slatd, nsMax * sizeof(float));
    cudaMalloc((void**)&pix, nsMax * sizeof(float2));

    // Loop over pixel chunks.
    for (chunk = 0; chunk < chunks; ++chunk) {
        const int srcStart = chunk * nsMax;
        int srcInBlock = ns - srcStart;
        if (srcInBlock > nsMax) srcInBlock = nsMax;

        // Copy test source positions for this chunk to the device.
        cudaMemcpy(slond, slon + srcStart, srcInBlock * sizeof(float),
                cudaMemcpyHostToDevice);
        cudaMemcpy(slatd, slat + srcStart, srcInBlock * sizeof(float),
                cudaMemcpyHostToDevice);

        // Invoke kernel to compute the (partial) beam pattern on the device.
        int threadsPerBlock = 256;
        int blocks = (srcInBlock + threadsPerBlock - 1) / threadsPerBlock;
        int maxAntennasPerBlock = 864; // Should be multiple of 16.
        size_t sharedMem = (threadsPerBlock + 2 * maxAntennasPerBlock)
                    * sizeof(float2);
        _beamPattern2dHorizontalWeights <<<blocks, threadsPerBlock, sharedMem>>>
                (na, axd, ayd, weights, srcInBlock, slond, slatd, k,
                        maxAntennasPerBlock, pix);
        cudaThreadSynchronize();
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess)
            printf("CUDA Error: %s\n", cudaGetErrorString(err));

        // Copy (partial) result from device memory to host memory.
        cudaMemcpy(image + 2 * srcStart, pix, srcInBlock * sizeof(float2),
                cudaMemcpyDeviceToHost);
    }

    // Free device memory.
    cudaFree(axd);
    cudaFree(ayd);
    cudaFree(weights);
    cudaFree(slond);
    cudaFree(slatd);
    cudaFree(pix);
    cudaFree(sbad);
    cudaFree(cbad);
    cudaFree(cbed);
}

#ifdef __cplusplus
}
#endif
