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

#include "cuda/oskar_cuda_hbp2higu.h"
#include "cuda/kernels/oskar_cudak_bp2hiw.h"
#include "cuda/kernels/oskar_cudak_bp2hiws.h"
#include "cuda/kernels/oskar_cudak_wt2hg.h"
#include <stdio.h>

#include "cuda/CudaEclipse.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_cuda_hbp2higu(int n2, int* n1, const float* x1, const float* y1,
        const float* x2, const float* y2, int ns, const float* sa,
        const float* se, float ba1, float be1, float ba2, float be2,
        float k, float* image)
{
    // Find total number of antennas.
    int na = 0, maxTileSize = 0;
    for (int i = 0; i < n2; ++i) {
        na += n1[i];
        maxTileSize = (maxTileSize > n1[i]) ? maxTileSize : n1[i];
    }
    int maxWeights = (maxTileSize > n2) ? maxTileSize : n2;

    // Precompute.
    float3 trig1 = make_float3(cos(ba1), sin(ba1), cos(be1));
    float3 trig2 = make_float3(cos(ba2), sin(ba2), cos(be2));

    // Allocate memory for antenna positions, antenna weights,
    // test source positions and pixel values on the device.
    float *x1d, *y1d, *x2d, *y2d, *sad, *sed;
    float2 *weights, *imaged, *signalsd;
    float3 *trig1d, *trig2d;
    cudaMalloc((void**)&x1d, na * sizeof(float));
    cudaMalloc((void**)&y1d, na * sizeof(float));
    cudaMalloc((void**)&x2d, n2 * sizeof(float));
    cudaMalloc((void**)&y2d, n2 * sizeof(float));
    cudaMalloc((void**)&weights, maxWeights * sizeof(float2));
    cudaMalloc((void**)&trig1d, 1 * sizeof(float3));
    cudaMalloc((void**)&trig2d, 1 * sizeof(float3));

    // Copy antenna positions and beam geometry to device.
    cudaMemcpy(x1d, x1, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y1d, y1, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(x2d, x2, n2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y2d, y2, n2 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(trig1d, &trig1, 1 * sizeof(float3), cudaMemcpyHostToDevice);
    cudaMemcpy(trig2d, &trig2, 1 * sizeof(float3), cudaMemcpyHostToDevice);

    // Divide up the source (pixel) list into manageable chunks.
    int nsMax = 100000;
    int chunk = 0, chunks = (ns + nsMax - 1) / nsMax;

    // Allocate memory for source position chunk on the device.
    cudaMalloc((void**)&sad, nsMax * sizeof(float));
    cudaMalloc((void**)&sed, nsMax * sizeof(float));
    cudaMalloc((void**)&imaged, nsMax * sizeof(float2));
    cudaMalloc((void**)&signalsd, n2 * nsMax * sizeof(float2));
    cudaThreadSynchronize();
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        printf("CUDA Error (malloc): %s\n", cudaGetErrorString(err));

    // Loop over pixel chunks.
    for (chunk = 0; chunk < chunks; ++chunk) {

        const int srcStart = chunk * nsMax;
        int srcInBlock = ns - srcStart;
        if (srcInBlock > nsMax) srcInBlock = nsMax;

        // Copy test source positions for this chunk to the device.
        cudaMemcpy(sad, sa + srcStart, srcInBlock * sizeof(float),
                cudaMemcpyHostToDevice);
        cudaMemcpy(sed, se + srcStart, srcInBlock * sizeof(float),
                cudaMemcpyHostToDevice);

        // Loop over tiles.
        int antCount = 0;
        for (int t = 0; t < n2; ++t) {

            // Get number of antennas in the tile.
            int nat = n1[t];
            float* x1td = x1d + antCount;
            float* y1td = y1d + antCount;
            float2* signalstd = signalsd + t * srcInBlock;

            // Invoke kernel to compute antenna weights on the device.
            dim3 wThreads(256, 1);
            dim3 wBlocks((nat + wThreads.x - 1) / wThreads.x, 1);
            size_t wSharedMem = wThreads.x * sizeof(float2) + sizeof(float3);
            oskar_cudak_wt2hg <<<wBlocks, wThreads, wSharedMem>>> (
                    nat, x1td, y1td, 1, trig1d, k, weights);
            cudaThreadSynchronize();
            cudaError_t err = cudaPeekAtLastError();
            if (err != cudaSuccess)
                printf("CUDA Error (weights): %s\n", cudaGetErrorString(err));

            // Invoke kernel to compute the (partial) beam pattern on the device.
            int threadsPerBlock = 256;
            int blocks = (srcInBlock + threadsPerBlock - 1) / threadsPerBlock;
            int maxAntennasPerBlock = 864; // Should be multiple of 16.
            size_t sharedMem = 2 * maxAntennasPerBlock * sizeof(float2);
            oskar_cudak_bp2hiw <<<blocks, threadsPerBlock, sharedMem>>>
                    (nat, x1td, y1td, weights, srcInBlock, sad, sed, k,
                            maxAntennasPerBlock, signalstd);
            cudaThreadSynchronize();
            err = cudaPeekAtLastError();
            if (err != cudaSuccess)
                printf("CUDA Error (bp): %s\n", cudaGetErrorString(err));

            // Increment by number of antennas in tile.
            antCount += nat;
        }

        // Invoke kernel to compute tile weights on the device.
        dim3 wThreads(256, 1);
        dim3 wBlocks((n2 + wThreads.x - 1) / wThreads.x, 1);
        size_t wSharedMem = wThreads.x * sizeof(float2) + sizeof(float3);
        oskar_cudak_wt2hg <<<wBlocks, wThreads, wSharedMem>>> (
                n2, x2d, y2d, 1, trig2d, k, weights);
        cudaThreadSynchronize();

        // Beam pattern kernel that takes tile signals.
        // Invoke kernel to compute the (partial) beam pattern on the device.
        int threadsPerBlock = 256;
        int blocks = (srcInBlock + threadsPerBlock - 1) / threadsPerBlock;
        int maxAntennasPerBlock = 512; // Should be multiple of 16.
        size_t sharedMem = 3 * maxAntennasPerBlock * sizeof(float2);
        oskar_cudak_bp2hiws <<<blocks, threadsPerBlock, sharedMem>>>
                (n2, x2d, y2d, weights, signalsd, srcInBlock, srcInBlock, sad, sed, k,
                        maxAntennasPerBlock, imaged);
        cudaThreadSynchronize();
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess)
            printf("CUDA Error: %s\n", cudaGetErrorString(err));

        // Copy (partial) result from device memory to host memory.
        cudaMemcpy(image + 2 * srcStart, imaged, srcInBlock * sizeof(float2),
                cudaMemcpyDeviceToHost);
    }

    // Free device memory.
    cudaFree(x1d);
    cudaFree(y1d);
    cudaFree(x2d);
    cudaFree(y2d);
    cudaFree(weights);
    cudaFree(sad);
    cudaFree(sed);
    cudaFree(imaged);
    cudaFree(trig1d);
    cudaFree(trig2d);
    cudaFree(signalsd);
}

#ifdef __cplusplus
}
#endif
