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

#include "cuda/beamformer2dHorizontalIsotropicGeometric.h"
#include "cuda/_precompute2dHorizontalTrig.h"
#include "cuda/_antennaSignal2dHorizontalIsotropic.h"
#include "cuda/_weights2dHorizontalGeometric.h"

#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>

#include "cuda/CudaEclipse.h"
#include "cuda/CudaTypes.h"

#define TIMER_ENABLE 1
#include "utility/timer.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @details
 * Computes beams using CUDA.
 *
 * The computed beams are returned in the \p beams array, which
 * must be pre-sized to length 2*nb. The values in the \p beams array
 * are alternate (real, imag) pairs for each beam.
 *
 * @param[in] na The number of antennas.
 * @param[in] ax The antenna x-positions in metres.
 * @param[in] ay The antenna y-positions in metres.
 * @param[in] ns The number of sources.
 * @param[in] samp The source amplitudes.
 * @param[in] slon The source longitude coordinates in radians.
 * @param[in] slat The source latitude coordinates in radians.
 * @param[in] nb The number of beams to form.
 * @param[in] blon The source longitude coordinates in radians.
 * @param[in] blat The source latitude coordinates in radians.
 * @param[in] k The wavenumber (rad / m).
 * @param[out] beams The complex vector of output beams (length nb).
 */
void beamformer2dHorizontalIsotropicGeometric(const int na,
        const float* ax, const float* ay, const int ns, const float* samp,
        const float* slon, const float* slat, const int nb,
        const float* blon, const float* blat, const float k, float* beams)
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
    _precompute2dHorizontalTrig <<<sBlocks, threadsPerBlock>>>
            (ns, sposd, strigd);

    // Invoke kernel to compute antenna signals on the device.
    int aBlocks = (na + threadsPerBlock - 1) / threadsPerBlock;
    int maxSourcesPerBlock = 384;
    size_t aSharedMem = threadsPerBlock * sizeof(float2)
            + maxSourcesPerBlock * sizeof(float4);
    _antennaSignal2dHorizontalIsotropic <<<aBlocks,
            threadsPerBlock, aSharedMem>>>
            (na, axd, ayd, ns, sampd, strigd, k, maxSourcesPerBlock, signalsd);

    // Start beamforming loop.
    // There may not be enough memory to allocate a weights matrix big enough,
    // so we divide it up and only compute (at most) maxBeams at once.
    int block = 0, blocks = (nb + maxBeams - 1) / maxBeams;
    for (block = 0; block < blocks; ++block) {
        const int beamStart = block * maxBeams;
        int beamsInBlock = nb - beamStart;
        if (beamsInBlock > maxBeams) {
            beamsInBlock = maxBeams;
        }

        // Invoke kernel to precompute the beam positions on the device.
        int bBlocks = (beamsInBlock + threadsPerBlock - 1) / threadsPerBlock;
        _precompute2dHorizontalTrig <<<bBlocks, threadsPerBlock>>>
                (beamsInBlock, &bposd[beamStart], btrigd);

        // Invoke kernel to compute beamforming weights on the device.
        int wBlocks = (na*beamsInBlock + threadsPerBlock - 1) / threadsPerBlock;
        TIMER_START
        _weights2dHorizontalGeometric <<<wBlocks, threadsPerBlock>>> (
                na, axd, ayd, beamsInBlock, btrigd, k, weightsd);
        cudaThreadSynchronize();
        TIMER_STOP("Finished weights")

        // Call cuBLAS function to perform the matrix-vector multiplication.
        // Note that cuBLAS calls use Fortran-ordering (column major) for their
        // matrices, so we use the transpose here.
        TIMER_START
        cublasCgemv('t', na, beamsInBlock, make_float2(1.0, 0.0),
                weightsd, na, signalsd, 1, make_float2(0.0, 0.0), beamsd, 1);
        cudaThreadSynchronize();
        TIMER_STOP("Finished matrix-vector")

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
