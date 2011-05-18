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

#include "cuda/oskar_cuda_bp2hc.h"
#include "cuda/kernels/oskar_cudak_antenna.h"
#include "cuda/kernels/oskar_cudak_apodisation.h"
#include "cuda/kernels/oskar_cudak_bp2hiw.h"
#include "cuda/kernels/oskar_cudak_wt2hg.h"
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "cuda/CudaEclipse.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.

int oskar_cudaf_bp2hc(int na, const float* ax, const float* ay,
        int ns, const float* sa, const float* se, float ba, float be, float k,
        int apfn, int atype, float gain, float fwhm, float* image)
{
    // Initialise.
    cudaError_t errCuda = cudaSuccess;
    int i, a, csize, retVal = 0;
    const int nb = 1; // Number of beams is 1, since this is a beam pattern.
    const int naMax = 864; // Should be multiple of 16.
    const int nsMax = 100000; // Maximum number of sources per iteration.
    float r, rmax = 0.0f; // Station radius.

    // Set up thread blocks.
    const dim3 wThd(256, 1); // Weights generator (antennas, beams).
    const dim3 wBlk((na + wThd.x - 1) / wThd.x, 1);
    const size_t wSmem = wThd.x * sizeof(float2) + sizeof(float3);
    const int aThd = 256; // Apodisation function generator (antennas).
    const int aBlk = (na + aThd - 1) / aThd;
    const int bThd = 256; // Beam pattern generator (source positions).
    int bBlk = 0; // Number of thread blocks for beam pattern computed later.
    size_t bSmem = 2 * naMax * sizeof(float2);

    // Precompute beam geometry.
    const float3 trig = make_float3(cos(ba), sin(ba), cos(be));

    // Allocate memory for antenna positions, antenna weights,
    // test source position chunk and pixel value chunk on the device.
    float *axd, *ayd, *sad, *sed;
    float2 *wts, *imaged;
    float3 *trigd;
    cudaMalloc((void**)&axd, na * sizeof(float));
    cudaMalloc((void**)&ayd, na * sizeof(float));
    cudaMalloc((void**)&wts, na * sizeof(float2));
    cudaMalloc((void**)&trigd, nb * sizeof(float3));
    cudaMalloc((void**)&sad, nsMax * sizeof(float));
    cudaMalloc((void**)&sed, nsMax * sizeof(float));
    cudaMalloc((void**)&imaged, nsMax * sizeof(float2));

    // Copy antenna positions and beam geometry to device.
    cudaMemcpy(axd, ax, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ayd, ay, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(trigd, &trig, nb * sizeof(float3), cudaMemcpyHostToDevice);

    // Invoke kernel to compute unnormalised, geometric antenna weights.
    oskar_cudakf_wt2hg <<< wBlk, wThd, wSmem >>>
            (na, axd, ayd, nb, trigd, k, wts);
    cudaThreadSynchronize();
    errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) goto stop;

    // Weights apodisation.
    if (apfn != apfn_none)
    {
        // Determine station radius.
        for (a = 0; a < na; ++a)
        {
            r = sqrtf(ax[a] * ax[a] + ay[a] * ay[a]);
            if (r > rmax) rmax = r;
        }

        // Switch on selected apodisation function.
        switch (apfn)
        {
        case apfn_hann:
            oskar_cudakf_apodisation_hann <<< aBlk, aThd >>> (
                    na, axd, ayd, nb, rmax, wts);
            break;
        default:
            printf("ERROR: Unknown apodisation function.\n");
            retVal = EXIT_FAILURE;
            goto stop;
        }
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) goto stop;
    }

    // Iterate over pixel chunks.
    for (i = 0; i < ns; i += nsMax)
    {
        csize = ns - i; // Chunk size.
        if (csize > nsMax) csize = nsMax;

        // Copy test source positions for this chunk to the device.
        cudaMemcpy(sad, sa + i, csize * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(sed, se + i, csize * sizeof(float), cudaMemcpyHostToDevice);

        // Invoke kernel to compute the (partial) beam pattern on the device.
        bBlk = (csize + bThd - 1) / bThd;
        oskar_cudakf_bp2hiw <<< bBlk, bThd, bSmem >>>
                (na, axd, ayd, wts, csize, sad, sed, k, naMax, imaged);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) goto stop;

        // Antenna type.
        if (atype != atype_isotropic)
        {
            // Switch on selected antenna type.
            switch (atype)
            {
            case atype_sin_elevation:
                oskar_cudakf_antenna_sine <<< bBlk, bThd >>> (
                        csize, sed, imaged);
                break;
            case atype_sin2_elevation:
                oskar_cudakf_antenna_sine_squared <<< bBlk, bThd >>> (
                        csize, sed, imaged);
                break;
            case atype_gaussian:
                oskar_cudakf_antenna_gaussian <<< bBlk, bThd >>> (
                        csize, sed, gain, fwhm, imaged);
                break;
            default:
                printf("ERROR: Unknown antenna type.\n");
                retVal = EXIT_FAILURE;
                goto stop;
            }
            cudaThreadSynchronize();
            errCuda = cudaPeekAtLastError();
            if (errCuda != cudaSuccess) goto stop;
        }

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
    cudaFree(wts);
    cudaFree(sad);
    cudaFree(sed);
    cudaFree(imaged);
    cudaFree(trigd);

    return retVal;
}

// Double precision.

int oskar_cudad_bp2hc(int na, const double* ax, const double* ay,
        int ns, const double* sa, const double* se, double ba, double be,
        double k, int apfn, int atype, double gain, double fwhm, double* image)
{
    // Initialise.
    cudaError_t errCuda = cudaSuccess;
    int i, a, csize, retVal = 0;
    const int nb = 1; // Number of beams is 1, since this is a beam pattern.
    const int naMax = 432; // Should be multiple of 16.
    const int nsMax = 100000; // Maximum number of sources per iteration.
    double r, rmax = 0.0; // Station radius.

    // Set up thread blocks.
    const dim3 wThd(256, 1); // Weights generator (antennas, beams).
    const dim3 wBlk((na + wThd.x - 1) / wThd.x, 1);
    const size_t wSmem = wThd.x * sizeof(double2) + sizeof(double3);
    const int aThd = 256; // Apodisation function generator (antennas).
    const int aBlk = (na + aThd - 1) / aThd;
    const int bThd = 256; // Beam pattern generator (source positions).
    int bBlk = 0; // Number of thread blocks for beam pattern computed later.
    size_t bSmem = 2 * naMax * sizeof(double2);

    // Precompute beam geometry.
    const double3 trig = make_double3(cos(ba), sin(ba), cos(be));

    // Allocate memory for antenna positions, antenna weights,
    // test source position chunk and pixel value chunk on the device.
    double *axd, *ayd, *sad, *sed;
    double2 *wts, *imaged;
    double3 *trigd;
    cudaMalloc((void**)&axd, na * sizeof(double));
    cudaMalloc((void**)&ayd, na * sizeof(double));
    cudaMalloc((void**)&wts, na * sizeof(double2));
    cudaMalloc((void**)&trigd, nb * sizeof(double3));
    cudaMalloc((void**)&sad, nsMax * sizeof(double));
    cudaMalloc((void**)&sed, nsMax * sizeof(double));
    cudaMalloc((void**)&imaged, nsMax * sizeof(double2));

    // Copy antenna positions and beam geometry to device.
    cudaMemcpy(axd, ax, na * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ayd, ay, na * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(trigd, &trig, nb * sizeof(double3), cudaMemcpyHostToDevice);

    // Invoke kernel to compute unnormalised, geometric antenna weights.
    oskar_cudakd_wt2hg <<< wBlk, wThd, wSmem >>>
            (na, axd, ayd, nb, trigd, k, wts);
    cudaThreadSynchronize();
    errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) goto stop;

    // Weights apodisation.
    if (apfn != apfn_none)
    {
        // Determine station radius.
        for (a = 0; a < na; ++a)
        {
            r = sqrt(ax[a] * ax[a] + ay[a] * ay[a]);
            if (r > rmax) rmax = r;
        }

        // Switch on selected apodisation function.
        switch (apfn)
        {
        case apfn_hann:
            oskar_cudakd_apodisation_hann <<< aBlk, aThd >>> (
                    na, axd, ayd, nb, rmax, wts);
            break;
        default:
            printf("ERROR: Unknown apodisation function.\n");
            retVal = EXIT_FAILURE;
            goto stop;
        }
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) goto stop;
    }

    // Iterate over pixel chunks.
    for (i = 0; i < ns; i += nsMax)
    {
        csize = ns - i; // Chunk size.
        if (csize > nsMax) csize = nsMax;

        // Copy test source positions for this chunk to the device.
        cudaMemcpy(sad, sa + i, csize * sizeof(double), cudaMemcpyHostToDevice);
        cudaMemcpy(sed, se + i, csize * sizeof(double), cudaMemcpyHostToDevice);

        // Invoke kernel to compute the (partial) beam pattern on the device.
        bBlk = (csize + bThd - 1) / bThd;
        oskar_cudakd_bp2hiw <<< bBlk, bThd, bSmem >>>
                (na, axd, ayd, wts, csize, sad, sed, k, naMax, imaged);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) goto stop;

        // Antenna type.
        if (atype != atype_isotropic)
        {
            // Switch on selected antenna type.
            switch (atype)
            {
            case atype_sin_elevation:
                oskar_cudakd_antenna_sine <<< bBlk, bThd >>> (
                        csize, sed, imaged);
                break;
            case atype_sin2_elevation:
                oskar_cudakd_antenna_sine_squared <<< bBlk, bThd >>> (
                        csize, sed, imaged);
                break;
            case atype_gaussian:
                oskar_cudakd_antenna_gaussian <<< bBlk, bThd >>> (
                        csize, sed, gain, fwhm, imaged);
                break;
            default:
                printf("ERROR: Unknown antenna type.\n");
                retVal = EXIT_FAILURE;
                goto stop;
            }
            cudaThreadSynchronize();
            errCuda = cudaPeekAtLastError();
            if (errCuda != cudaSuccess) goto stop;
        }

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
    cudaFree(wts);
    cudaFree(sad);
    cudaFree(sed);
    cudaFree(imaged);
    cudaFree(trigd);

    return retVal;
}

#ifdef __cplusplus
}
#endif
