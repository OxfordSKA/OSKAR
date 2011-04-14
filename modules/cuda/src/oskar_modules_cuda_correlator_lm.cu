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

#include "cuda/oskar_cuda_rpw3leg.h"
#include "cuda/kernels/oskar_cudak_rpw3leglm.h"
#include "cuda/kernels/oskar_cudak_cmatadd.h"
#include "cuda/kernels/oskar_cudak_cmatmul.h"
#include "cuda/kernels/oskar_cudak_cmatset.h"
#include "math/synthesis/oskar_math_synthesis_xyz2uvw.h"
#include <stdlib.h>
#include <stdio.h>
#include <cublas.h>

#include "cuda/CudaEclipse.h"

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

int oskar_modules_cuda_correlator_lm(int na, const float* ax, const float* ay,
        const float* az, int ns, const float* l, const float* m,
        const float* bsqrt, const float* e, float ra0, float dec0,
        float lst0, int nsdt, float sdt, float k, float* vis, float* u,
        float* v, float* w)
{
    // Initialise.
    cudaError_t err = cudaSuccess;
    cublasStatus cubError;
    float lst, ha0;
    float2 czero = make_float2(0.0, 0.0);
    int i, a;
    cublasInit();

    // Set up thread blocks.
    dim3 kThd(64, 4); // Sources, antennas.
    dim3 kBlk((ns + kThd.x - 1) / kThd.x, (na + kThd.y - 1) / kThd.y);
    size_t sMem = kThd.x * sizeof(float3);
    dim3 mThd(64, 4); // Sources, antennas.
    dim3 mBlk((ns + mThd.x - 1) / mThd.x, (na + mThd.y - 1) / mThd.y);
    dim3 vThd(16, 16); // Antennas, antennas.
    dim3 vBlk((na + vThd.x - 1) / vThd.x, (na + vThd.y - 1) / vThd.y);

    // Compute the source n-coordinates from l and m.
    float* n = (float*)malloc(ns * sizeof(float));
    for (i = 0; i < ns; ++i)
    {
        n[i] = sqrt(1.0f - l[i]*l[i] - m[i]*m[i]) - 1.0f;
    }

    // Scale source brightnesses (in Bsqrt) by station beams (in E).
    float* eb = (float*)malloc(ns * na * sizeof(float2));
    for (a = 0; a < na; ++a)
    {
        for (i = 0; i < ns; ++i)
        {
            int idx = i + a * ns;
            float bs = bsqrt[i];
            eb[idx + 0] = e[idx + 0] * bs; // Real
            eb[idx + 1] = e[idx + 1] * bs; // Imag
        }
    }

    // Allocate host memory for station u,v,w coordinates.
    float* uvw = (float*)malloc(na * 3 * sizeof(float));

    // Allocate memory for source coordinates and visibility matrix on the
    // device.
    float *ld, *md, *nd;
    float2 *visd, *visw, *kmat, *emat;
    cudaMalloc((void**)&ld, ns * sizeof(float));
    cudaMalloc((void**)&md, ns * sizeof(float));
    cudaMalloc((void**)&nd, ns * sizeof(float));
    cudaMalloc((void**)&visd, na * na * sizeof(float2));
    cudaMalloc((void**)&visw, na * na * sizeof(float2));
    cudaMalloc((void**)&kmat, ns * na * sizeof(float2));
    cudaMalloc((void**)&emat, ns * na * sizeof(float2));
    cudaThreadSynchronize();
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) goto stop;

    // Copy source coordinates to device.
    cudaMemcpy(ld, l, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(md, m, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(nd, n, ns * sizeof(float), cudaMemcpyHostToDevice);

    // Copy modified E-matrix to device.
    cudaMemcpy(emat, eb, ns * na * sizeof(float2), cudaMemcpyHostToDevice);

    // Clear visibility matrix.
    oskar_cudak_cmatset <<<vBlk, vThd>>> (na, na, czero, visd);
    cudaThreadSynchronize();
    err = cudaPeekAtLastError();
    if (err != cudaSuccess) goto stop;

    // Loop over integrations.
    for (i = 0; i < nsdt; ++i)
    {
        // Compute the current LST and hour angle of the phase centre.
        lst = lst0 + 2 * M_PI * i * sdt / 86400.0f;
        ha0 = lst - ra0;

        // Compute the station u,v,w coordinates.
        oskar_math_synthesis_xyz2uvw(na, ax, ay, az, ha0, dec0,
                &uvw[0], &uvw[na], &uvw[2*na]);

        // Copy u,v,w coordinates to constant memory.
        cudaMemcpyToSymbol(uvwd, uvw, na * 3 * sizeof(float));
        cudaThreadSynchronize();
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) goto stop;

        // Compute K-matrix.
        oskar_cudak_rpw3leglm <<<kBlk, kThd, sMem>>> (
                na, ns, ld, md, nd, k, kmat);
        cudaThreadSynchronize();
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) goto stop;

        // Perform complex matrix element multiply.
        oskar_cudak_cmatmul <<<mBlk, mThd>>> (
                ns, na, kmat, emat, kmat);
        cudaThreadSynchronize();
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) goto stop;

        // Perform matrix-matrix multiply (reduction).
        cublasCgemm('c', 'n', na, na, ns, make_float2(1.0, 0.0),
                kmat, ns, kmat, ns, make_float2(0.0, 0.0), visw, na);
        cubError = cublasGetError();
        if (cubError != CUBLAS_STATUS_SUCCESS) {
            err = (cudaError_t)cubError;
            goto stop;
        }

        // Accumulate visibility matrix.
        oskar_cudak_cmatadd <<<vBlk, vThd>>> (ns, na, visd, visw, visd);
        cudaThreadSynchronize();
        err = cudaPeekAtLastError();
        if (err != cudaSuccess) goto stop;
    }

    // Scale result.
    cublasCscal(na * na, make_float2(1.0 / nsdt, 0.0), visd, 1);
    cubError = cublasGetError();
    if (cubError != CUBLAS_STATUS_SUCCESS) {
        err = (cudaError_t)cubError;
        goto stop;
    }

    // Copy result to host.
    cudaMemcpy(vis, visd, na * na * sizeof(float2), cudaMemcpyDeviceToHost);

    // Clean up before exit.
    stop:
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Free host memory.
    free(eb);
    free(uvw);
    free(n);

    // Free device memory.
    cudaFree(kmat);
    cudaFree(emat);
    cudaFree(ld);
    cudaFree(md);
    cudaFree(nd);
    cudaFree(visd);
    cudaFree(visw);

    // Shutdown.
    cublasShutdown();
    return err;
}

#ifdef __cplusplus
}
#endif
