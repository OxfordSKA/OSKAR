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

#include "cuda/oskar_cuda_bp2phc.h"
#include "cuda/kernels/oskar_cudak_bp2phiw.h"
#include "cuda/kernels/oskar_cudak_wt2phg.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.

int oskar_cudaf_bp2phc(int na, const float* ax, const float* ay,
        int ns, const float* scace, const float* ssace, float ba, float be,
        float* image)
{
    // Initialise.
    cudaError_t errCuda = cudaSuccess;
    int i, csize, retVal = 0;
    const int nb = 1; // Number of beams is 1, since this is a beam pattern.
    const int naMax = 864; // Should be multiple of 16.
    const int nsMax = 100000; // Maximum number of sources per iteration.

    // Set up thread blocks.
    const dim3 wThd(256, 1); // Weights generator (antennas, beams).
    const dim3 wBlk((na + wThd.x - 1) / wThd.x, 1);
    const size_t wSmem = wThd.x * sizeof(float2) + sizeof(float3);
    const int bThd = 256; // Beam pattern generator (source positions).
    int bBlk = 0; // Number of thread blocks for beam pattern computed later.
    size_t bSmem = 2 * naMax * sizeof(float2);

    // Compute beam geometry.
    const float bcace = cos(ba) * cos(be);
    const float bsace = sin(ba) * cos(be);

    // Allocate memory for antenna positions, antenna weights,
    // test source position chunk and pixel value chunk on the device.
    float *axd, *ayd, *bcaced, *bsaced, *scaced, *ssaced;
    float2 *wts, *imaged;
    cudaMalloc((void**)&axd, na * sizeof(float));
    cudaMalloc((void**)&ayd, na * sizeof(float));
    cudaMalloc((void**)&bcaced, nb * sizeof(float));
    cudaMalloc((void**)&bsaced, nb * sizeof(float));
    cudaMalloc((void**)&scaced, nsMax * sizeof(float));
    cudaMalloc((void**)&ssaced, nsMax * sizeof(float));
    cudaMalloc((void**)&wts, na * sizeof(float2));
    cudaMalloc((void**)&imaged, nsMax * sizeof(float2));

    // Copy antenna positions and beam geometry to device.
    cudaMemcpy(axd, ax, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ayd, ay, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bcaced, &bcace, nb * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bsaced, &bsace, nb * sizeof(float), cudaMemcpyHostToDevice);

    // Invoke kernel to compute unnormalised, geometric antenna weights.
    oskar_cudakf_wt2phg <<< wBlk, wThd, wSmem >>>
            (na, axd, ayd, nb, bcaced, bsaced, wts);
    cudaThreadSynchronize();
    errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) goto stop;

    // Iterate over pixel chunks.
    for (i = 0; i < ns; i += nsMax)
    {
        csize = ns - i; // Chunk size.
        if (csize > nsMax) csize = nsMax;

        // Copy test source positions for this chunk to the device.
        cudaMemcpy(scaced, scace + i, csize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ssaced, ssace + i, csize * sizeof(float), cudaMemcpyHostToDevice);

        // Invoke kernel to compute the (partial) beam pattern on the device.
        bBlk = (csize + bThd - 1) / bThd;
        oskar_cudakf_bp2phiw <<< bBlk, bThd, bSmem >>>
                (na, axd, ayd, wts, csize, scaced, ssaced, naMax, imaged);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) goto stop;

        // Copy (partial) result from device memory to host memory.
        cudaMemcpy(image + 2 * i, imaged, csize * sizeof(float2),
                cudaMemcpyDeviceToHost);
    }

    // Clean up before exit.
    stop:
    if (errCuda != cudaSuccess)
    {
        retVal = errCuda;
        printf("CUDA Error: %s\n", cudaGetErrorString(errCuda));
    }

    // Free device memory.
    cudaFree(axd);
    cudaFree(ayd);
    cudaFree(bcaced);
    cudaFree(bsaced);
    cudaFree(scaced);
    cudaFree(ssaced);
    cudaFree(wts);
    cudaFree(imaged);

    return retVal;
}

// Double precision.

int oskar_cudad_bp2phc(int na, const double* ax, const double* ay,
        int ns, const double* scace, const double* ssace, double ba, double be,
        double* image)
{
    // Initialise.
    cudaError_t errCuda = cudaSuccess;
    int i, csize, retVal = 0;
    const int nb = 1; // Number of beams is 1, since this is a beam pattern.
    const int naMax = 432; // Should be multiple of 16.
    const int nsMax = 100000; // Maximum number of sources per iteration.

    // Set up thread blocks.
    const dim3 wThd(256, 1); // Weights generator (antennas, beams).
    const dim3 wBlk((na + wThd.x - 1) / wThd.x, 1);
    const size_t wSmem = wThd.x * sizeof(double2) + sizeof(double3);
    const int bThd = 256; // Beam pattern generator (source positions).
    int bBlk = 0; // Number of thread blocks for beam pattern computed later.
    size_t bSmem = 3 * naMax * sizeof(double2);

    // Compute beam geometry.
    const double bcace = cos(ba) * cos(be);
    const double bsace = sin(ba) * cos(be);

    // Allocate memory for antenna positions, antenna weights,
    // test source position chunk and pixel value chunk on the device.
    double *axd, *ayd, *bcaced, *bsaced, *scaced, *ssaced;
    double2 *wts, *imaged;
    cudaMalloc((void**)&axd, na * sizeof(double));
    cudaMalloc((void**)&ayd, na * sizeof(double));
    cudaMalloc((void**)&bcaced, nb * sizeof(double));
    cudaMalloc((void**)&bsaced, nb * sizeof(double));
    cudaMalloc((void**)&scaced, nsMax * sizeof(double));
    cudaMalloc((void**)&ssaced, nsMax * sizeof(double));
    cudaMalloc((void**)&wts, na * sizeof(double2));
    cudaMalloc((void**)&imaged, nsMax * sizeof(double2));

    // Copy antenna positions and beam geometry to device.
    cudaMemcpy(axd, ax, na * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ayd, ay, na * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(bcaced, &bcace, nb * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(bsaced, &bsace, nb * sizeof(double), cudaMemcpyHostToDevice);

    // Invoke kernel to compute unnormalised, geometric antenna weights.
    oskar_cudakd_wt2phg <<< wBlk, wThd, wSmem >>>
            (na, axd, ayd, nb, bcaced, bsaced, wts);
    cudaThreadSynchronize();
    errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) goto stop;

    // Iterate over pixel chunks.
    for (i = 0; i < ns; i += nsMax)
    {
        csize = ns - i; // Chunk size.
        if (csize > nsMax) csize = nsMax;

        // Copy test source positions for this chunk to the device.
        cudaMemcpy(scaced, scace + i, csize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(ssaced, ssace + i, csize * sizeof(double), cudaMemcpyHostToDevice);

        // Invoke kernel to compute the (partial) beam pattern on the device.
        bBlk = (csize + bThd - 1) / bThd;
        oskar_cudakd_bp2phiw <<< bBlk, bThd, bSmem >>>
                (na, axd, ayd, wts, csize, scaced, ssaced, naMax, imaged);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) goto stop;

        // Copy (partial) result from device memory to host memory.
        cudaMemcpy(image + 2 * i, imaged, csize * sizeof(double2),
                cudaMemcpyDeviceToHost);
    }

    // Clean up before exit.
    stop:
    if (errCuda != cudaSuccess)
    {
        retVal = errCuda;
        printf("CUDA Error: %s\n", cudaGetErrorString(errCuda));
    }

    // Free device memory.
    cudaFree(axd);
    cudaFree(ayd);
    cudaFree(bcaced);
    cudaFree(bsaced);
    cudaFree(scaced);
    cudaFree(ssaced);
    cudaFree(wts);
    cudaFree(imaged);

    return retVal;
}

#ifdef __cplusplus
}
#endif
