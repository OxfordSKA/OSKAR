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

#include "cuda/oskar_cuda_bp2hugg.h"
#include "cuda/kernels/oskar_cudak_bp2hugw.h"
#include "cuda/kernels/oskar_cudak_wt2hg.h"
#include <stdio.h>

#include "cuda/CudaEclipse.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_cuda_bp2hugg(int na, const float* ax, const float* ay,
        const float* aw, const float* ag, int ns, const float* slon,
        const float* slat, float ba, float be, float k, float* image)
{
    // Precompute.
    float3 trig = make_float3(cos(ba), sin(ba), cos(be));

    // Allocate memory for antenna data, antenna weights,
    // test source positions and pixel values on the device.
    float *axd, *ayd, *awd, *agd, *slond, *slatd;
    float2 *weights, *pix;
    float3 *trigd;
    cudaMalloc((void**)&axd, na * sizeof(float));
    cudaMalloc((void**)&ayd, na * sizeof(float));
    cudaMalloc((void**)&awd, na * sizeof(float));
    cudaMalloc((void**)&agd, na * sizeof(float));
    cudaMalloc((void**)&weights, na * sizeof(float2));
    cudaMalloc((void**)&trigd, 1 * sizeof(float3));

    // Precompute antenna beam width parameters.
    float* awh = (float*)malloc(na * sizeof(float));
    for (int a = 0; a < na; ++a) {
        awh[a] = 2.772588 / (aw[a] * aw[a]); // 4 ln2 / (FWHM^2)
    }

    // Copy antenna data and beam geometry to device.
    cudaMemcpy(axd, ax, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ayd, ay, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(awd, awh, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(agd, ag, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(trigd, &trig, 1 * sizeof(float3), cudaMemcpyHostToDevice);

    // Invoke kernel to compute unnormalised antenna weights on the device.
    dim3 wThreads(256, 1);
    dim3 wBlocks((na + wThreads.x - 1) / wThreads.x, 1);
    size_t wSharedMem = wThreads.x * sizeof(float2) + sizeof(float3);
    oskar_cudak_wt2hg <<<wBlocks, wThreads, wSharedMem>>> (
            na, axd, ayd, 1, trigd, k, weights);
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
        int maxAntennasPerBlock = 640; // Should be multiple of 16.
        size_t sharedMem = 3 * maxAntennasPerBlock * sizeof(float2);
        oskar_cudak_bp2hugw <<<blocks, threadsPerBlock, sharedMem>>>
                (na, axd, ayd, awd, agd, weights, srcInBlock, slond, slatd, k,
                        maxAntennasPerBlock, pix);
        cudaThreadSynchronize();
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess)
            printf("CUDA Error: %s\n", cudaGetErrorString(err));

        // Copy (partial) result from device memory to host memory.
        cudaMemcpy(image + 2 * srcStart, pix, srcInBlock * sizeof(float2),
                cudaMemcpyDeviceToHost);
    }

    // Free host memory.
    free(awh);

    // Free device memory.
    cudaFree(axd);
    cudaFree(ayd);
    cudaFree(awd);
    cudaFree(agd);
    cudaFree(weights);
    cudaFree(slond);
    cudaFree(slatd);
    cudaFree(pix);
    cudaFree(trigd);
}

#ifdef __cplusplus
}
#endif
