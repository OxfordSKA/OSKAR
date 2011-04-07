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

#include "cuda/oskar_cuda_bp2higa.h"
#include "cuda/kernels/oskar_cudak_bp2hiw.h"
#include "cuda/kernels/oskar_cudak_wt2hg.h"
#include "cuda/kernels/oskar_cudak_hann.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda/CudaEclipse.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_cuda_bp2higa(int na, const float* ax, const float* ay,
        int ns, const float* slon, const float* slat,
        float ba, float be, float k, int apFn, float* image)
{
    const int nb = 1; // Number of beams == 1 as this is a beam pattern!.

    // Precompute.
    float3 trig = make_float3(cos(ba), sin(ba), cos(be));

    // Allocate memory for antenna positions, antenna weights,
    // test source positions and pixel values on the device.
    float *axd, *ayd, *slond, *slatd;
    float2 *weights, *pix;
    float3 *trigd;
    cudaMalloc((void**)&axd, na * sizeof(float));
    cudaMalloc((void**)&ayd, na * sizeof(float));
    cudaMalloc((void**)&weights, na * sizeof(float2));
    cudaMalloc((void**)&trigd, 1 * sizeof(float3));

    // Copy antenna positions and beam geometry to device.
    cudaMemcpy(axd, ax, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ayd, ay, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(trigd, &trig, 1 * sizeof(float3), cudaMemcpyHostToDevice);

    // Invoke kernel to compute unnormalised antenna weights on the device.
    dim3 wThreads(256, 1);
    dim3 wBlocks((na + wThreads.x - 1) / wThreads.x, 1);
    size_t wSharedMem = wThreads.x * sizeof(float2) + sizeof(float3);
    oskar_cudak_wt2hg <<< wBlocks, wThreads, wSharedMem >>> (
            na, axd, ayd, nb, trigd, k, weights);
    cudaThreadSynchronize();

    // Invoke kernel to modify the weights by an apodistion function.
    float rMax = 0.0f;
    for (int a = 0; a < na; ++a)
    {
        const float r = sqrtf(ax[a] * ax[a] + ay[a] * ay[a]);
        if (r > rMax) rMax = r;
    }
    const int aThreads = 256;
    const int aBlocks = (int) ceil((float)na / (float)aThreads);
    switch (apFn)
    {
        case apFn_none:
            break;
        case apFn_hann:
        {
            const float fwhm = rMax;
            oskar_cudak_hann <<< aBlocks, aThreads >>> (na, axd, ayd, nb,
                    fwhm, weights);
            break;
        }
        default:
        {
            printf("ERROR: Undefined apodisation function.\n");
            return EXIT_FAILURE;
        }
    };


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
        size_t sharedMem = 2 * maxAntennasPerBlock * sizeof(float2);
        oskar_cudak_bp2hiw <<<blocks, threadsPerBlock, sharedMem>>>
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
    cudaFree(trigd);

    return EXIT_SUCCESS;
}

#ifdef __cplusplus
}
#endif
