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

#include "cuda/oskar_cuda_bf2hig.h"
#include "cuda/kernels/oskar_cudak_pc2ht.h"
#include "cuda/kernels/oskar_cudak_as2hi.h"
#include "cuda/kernels/oskar_cudak_wt2hg.h"

#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>

#include "cuda/CudaEclipse.h"

#define TIMER_ENABLE 1
#include "utility/timer.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_cudaf_bf2hig(int na, const float* ax, const float* ay,
        int ns, const float* samp, const float* slon, const float* slat,
        int nb, const float* blon, const float* blat, float k,
        float* beams)
{
    // Initialise cuBLAS.
    cublasInit();

    // Create source and beam position pairs in host memory.
    float2* spos = (float2*)calloc(ns, sizeof(float2));
    float2* bpos = (float2*)calloc(nb, sizeof(float2));
    int i;
    for (i = 0; i < ns; ++i)
        spos[i] = make_float2(slon[i], slat[i]);
    for (i = 0; i < nb; ++i)
        bpos[i] = make_float2(blon[i], blat[i]);

    // Allocate memory for antenna positions, source positions,
    // beam positions,
    // antenna signals, beamforming weights on the device.
    float *axd, *ayd, *sampd;
    float2 *sposd, *bposd, *signalsd, *weightsd, *beamsd;
    float3 *strigd, *btrigd;
    cudaMalloc((void**)&axd, na * sizeof(float));
    cudaMalloc((void**)&ayd, na * sizeof(float));
    cudaMalloc((void**)&sampd, ns * sizeof(float));
    cudaMalloc((void**)&sposd, ns * sizeof(float2));
    cudaMalloc((void**)&bposd, nb * sizeof(float2));
    cudaMalloc((void**)&strigd, ns * sizeof(float3));
    cudaMalloc((void**)&signalsd, na * sizeof(float2));

    // Copy antenna positions, source positions and beam positions to device.
    cudaMemcpy(axd, ax, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ayd, ay, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sampd, samp, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(sposd, &spos[0], ns * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(bposd, &bpos[0], nb * sizeof(float2), cudaMemcpyHostToDevice);

    // Set the maximum number of beams the device can compute at once.
    const int maxBeams = 1000;

    // Allocate enough memory for the beams and weights blocks.
    cudaMalloc((void**)&weightsd, na * maxBeams * sizeof(float2));
    cudaMalloc((void**)&btrigd, maxBeams * sizeof(float3));
    cudaMalloc((void**)&beamsd, maxBeams * sizeof(float2));

    // Set threads per block.
    int threadsPerBlock = 384;

    // Invoke kernel to precompute source positions on the device.
    int sBlocks = (ns + threadsPerBlock - 1) / threadsPerBlock;
    oskar_cudakf_pc2ht <<<sBlocks, threadsPerBlock>>>
            (ns, sposd, strigd);

    // Invoke kernel to compute antenna signals on the device.
    int aBlocks = (na + threadsPerBlock - 1) / threadsPerBlock;
    int maxSourcesPerBlock = 384;
    size_t aSharedMem = maxSourcesPerBlock * sizeof(float4);
    oskar_cudakf_as2hi <<<aBlocks,
            threadsPerBlock, aSharedMem>>>
            (na, axd, ayd, ns, sampd, strigd, k, maxSourcesPerBlock, signalsd);

    // Start beamforming loop.
    // There may not be enough memory to allocate a weights matrix big enough,
    // so we divide it up and only compute (at most) maxBeams at once.
    int block = 0, blocks = (nb + maxBeams - 1) / maxBeams;
    for (block = 0; block < blocks; ++block) {
        const int beamStart = block * maxBeams;
        int beamsInBlock = nb - beamStart;
        if (beamsInBlock > maxBeams)
            beamsInBlock = maxBeams;

        // Invoke kernel to precompute the beam positions on the device.
        int bBlocks = (beamsInBlock + threadsPerBlock - 1) / threadsPerBlock;
        oskar_cudakf_pc2ht <<<bBlocks, threadsPerBlock>>>
                (beamsInBlock, &bposd[beamStart], btrigd);

        // Invoke kernel to compute beamforming weights on the device.
        dim3 wThreads(16, 16); // Antennas, beams.
        dim3 wBlocks((na + wThreads.x - 1) / wThreads.x,
                (beamsInBlock + wThreads.y - 1) / wThreads.y);
        size_t wSharedMem = wThreads.x * sizeof(float2)
                + wThreads.y * sizeof(float3);
        oskar_cudakf_wt2hg <<<wBlocks, wThreads, wSharedMem>>> (
                na, axd, ayd, beamsInBlock, btrigd, k, weightsd);
        cudaThreadSynchronize();

        // Call cuBLAS function to perform the matrix-vector multiplication.
        // Note that cuBLAS calls use Fortran-ordering (column major) for their
        // matrices, so we use the transpose here.
        cublasCgemv('t', na, beamsInBlock, make_float2(1.0, 0.0),
                weightsd, na, signalsd, 1, make_float2(0.0, 0.0), beamsd, 1);
        cudaThreadSynchronize();

        // Copy result from device memory to host memory.
        cudaMemcpy(&beams[2*beamStart], beamsd, beamsInBlock * sizeof(float2),
                cudaMemcpyDeviceToHost);
    }

    // Free device memory.
    cudaFree(axd);
    cudaFree(ayd);
    cudaFree(sampd);
    cudaFree(sposd);
    cudaFree(strigd);
    cudaFree(btrigd);
    cudaFree(signalsd);
    cudaFree(weightsd);
    cudaFree(beamsd);

    // Free host memory.
    free(spos);
    free(bpos);

    // Shut down cuBLAS.
    cublasShutdown();
}

#ifdef __cplusplus
}
#endif
