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

#include "cuda/oskar_cuda_as2hi.h"
#include "cuda/kernels/oskar_cudak_as2hi.h"
#include "cuda/kernels/oskar_cudak_pc2ht.h"
#include <stdio.h>
#include <stdlib.h>

#include "cuda/CudaEclipse.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_cuda_as2hi(const int na, const float* ax, const float* ay,
        const int ns, const float* samp, const float* slon, const float* slat,
        const float k, float* signals)
{
    // Create source position pairs in host memory.
    float2* spos = (float2*)calloc(ns, sizeof(float2));
    int i = 0;
    for (i = 0; i < ns; ++i)
        spos[i] = make_float2(slon[i], slat[i]);

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
    cudaMemcpy(sposd, spos, ns * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(sampd, samp, ns * sizeof(float), cudaMemcpyHostToDevice);

    // Error code.
    cudaError_t err;

    // Invoke kernel to pre-compute source positions on the device.
    int sThreadsPerBlock = 384;
    int sBlocks = (ns + sThreadsPerBlock - 1) / sThreadsPerBlock;
    oskar_cudak_pc2ht <<<sBlocks, sThreadsPerBlock>>> (ns, sposd, strigd);
    err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Invoke kernel to compute antenna signals on the device.
    int threadsPerBlock = 384;
    int blocks = (na + threadsPerBlock - 1) / threadsPerBlock;
    int maxSourcesPerBlock = 384;
    size_t sharedMem = threadsPerBlock * sizeof(float2)
            + maxSourcesPerBlock * sizeof(float4);
    oskar_cudak_as2hi <<<blocks, threadsPerBlock, sharedMem>>>
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

    // Free host memory.
    free(spos);
}

#ifdef __cplusplus
}
#endif
