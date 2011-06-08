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

#include "modules/cuda/oskar_modules_cuda_correlator_lm_bw.h"

#include "cuda/kernels/oskar_cudak_dftw_3d_seq_out.h"
#include "cuda/kernels/oskar_cudak_mat_mul_cc.h"
#include "cuda/kernels/oskar_cudak_correlator.h"
#include "math/core/oskar_math_core_ctrimat.h"
#include "math/synthesis/oskar_math_synthesis_baselines.h"
#include "math/synthesis/oskar_math_synthesis_xyz2uvw.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <cublas.h>

#include "cuda/CudaEclipse.h"

#ifdef __cplusplus
extern "C" {
#endif

#define C_0 299792458.0
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif


// Single precision.

int oskar_modules_cudaf_correlator_lm_bw(int na, const float* ax,
        const float* ay, const float* az, int ns, const float* l,
        const float* m, const float* bsqrt, const float* e, float ra0,
        float dec0, float lst0, int nsdt, float sdt, float k, float bandwidth,
        float* vis, float* u, float* v, float* w)
{
    // Initialise.
    cudaError_t errCuda = cudaSuccess;
    cublasStatus errCublas = CUBLAS_STATUS_SUCCESS;
    double lst, ha0; // Must be double.
    float2 one = make_float2(1.0f, 0.0f);
    int i, a, retVal = 0;
    double tIncCentre, tInc;
    double tOffset = (double)sdt / 2.0;
    cublasInit();

    // Set up thread blocks.
    dim3 kThd(64, 4); // Sources, antennas.
    dim3 kBlk((ns + kThd.x - 1) / kThd.x, (na + kThd.y - 1) / kThd.y);
    size_t sMem = (kThd.x + kThd.y) * sizeof(float3);
    dim3 mThd(64, 4); // Sources, antennas.
    dim3 mBlk((ns + mThd.x - 1) / mThd.x, (na + mThd.y - 1) / mThd.y);
    dim3 vThd(256, 1); // Antennas, antennas.
    dim3 vBlk(na, na);
    size_t vsMem = vThd.x * sizeof(float2);

    // Get bandwidth term.
    float lambda_bandwidth = (2 * M_PI / k) * bandwidth;

    // Compute the source n-coordinates from l and m.
    float* n = (float*)malloc(ns * sizeof(float));
    float r = 0.0f;
    for (i = 0; i < ns; ++i)
    {
        r = l[i]*l[i] + m[i]*m[i];
        n[i] = (r < 1.0) ? sqrtf(1.0f - r) - 1.0f : -1.0f;
    }

    // Scale source brightnesses (in Bsqrt) by station beams (in E).
    float2* eb = (float2*)malloc(ns * na * sizeof(float2));
    for (a = 0; a < na; ++a)
    {
        for (i = 0; i < ns; ++i)
        {
            int idx = i + a * ns;
            float bs = bsqrt[i];
            eb[idx].x = e[2*idx + 0] * bs; // Real
            eb[idx].y = e[2*idx + 1] * bs; // Imag
        }
    }

    // Allocate host memory for station u,v,w coordinates and visibility matrix.
    int nb = na * (na - 1) / 2;
    float* uvw = (float*)malloc(na * 3 * sizeof(float));

    // Allocate memory for source coordinates and visibility matrix on the
    // device.
    float *ld, *md, *nd, *uvwd;
    float2 *visd, *kmat, *emat;
    cudaMalloc((void**)&ld, ns * sizeof(float));
    cudaMalloc((void**)&md, ns * sizeof(float));
    cudaMalloc((void**)&nd, ns * sizeof(float));
    cudaMalloc((void**)&visd, nb * sizeof(float2));
    cudaMalloc((void**)&kmat, ns * na * sizeof(float2));
    cudaMalloc((void**)&emat, ns * na * sizeof(float2));
    cudaMalloc((void**)&uvwd, 3 * na * sizeof(float));
    cudaThreadSynchronize();
    errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) goto stop;

    // Copy source coordinates and modified E-matrix to device.
    cudaMemcpy(ld, l, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(md, m, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(nd, n, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(emat, eb, ns * na * sizeof(float2), cudaMemcpyHostToDevice);

    // Clear visibility matrix.
    cudaMemset(visd, 0, nb * sizeof(float2));

    // Copy u,v,w baseline coordinates of mid-point to output arrays.
    // FIXME: probably don't need to return UVW from this function?
    tIncCentre = ((nsdt - 1) / 2) * sdt + tOffset;
    lst = lst0 + 2 * M_PI * tIncCentre * sdt / 86400.0f;
    ha0 = lst - ra0;
    oskar_math_synthesisf_xyz2uvw(na, ax, ay, az, ha0, dec0,
            &uvw[0], &uvw[na], &uvw[2*na]);
    oskar_math_synthesisf_baselines(na, &uvw[0], &uvw[na], &uvw[2*na],
            u, v, w);

    // Loop over integrations.
    for (i = 0; i < nsdt; ++i)
    {
        // Compute the current LST and hour angle of the phase centre.
        tInc = i * sdt + tOffset;
        lst = lst0 + 2 * M_PI * tInc / 86400.0;
        ha0 = lst - ra0;

        // Compute the station u,v,w coordinates.
        oskar_math_synthesisf_xyz2uvw(na, ax, ay, az, ha0, dec0,
                &uvw[0], &uvw[na], &uvw[2*na]);

        // Multiply station u,v,w coordinates by 2 pi / lambda.
        for (a = 0; a < na; ++a)
        {
            uvw[a] *= k;
            uvw[na + a] *= k;
            uvw[2 * na + a] *= k;
        }

        // Copy u,v,w coordinates to device.
        cudaMemcpy(uvwd, uvw, na * 3 * sizeof(float), cudaMemcpyHostToDevice);

        // Compute K-matrix.
        oskar_cudakf_dftw_3d_seq_out <<<kBlk, kThd, sMem>>> (
                na, &uvwd[0], &uvwd[na], &uvwd[2*na], ns, ld, md, nd, kmat);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) goto stop;

        // Perform complex matrix element multiply.
        oskar_cudakf_mat_mul_cc <<<mBlk, mThd>>> (ns, na, kmat, emat, kmat);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) goto stop;

        // Call the correlator kernel.
        oskar_cudakf_correlator <<<vBlk, vThd, vsMem>>> (
                ns, na, kmat, &uvwd[0], &uvwd[na], ld, md, lambda_bandwidth, visd);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) goto stop;
    }

    // Scale result.
    cublasCscal(nb, make_float2(1.0f / nsdt, 0.0f), visd, 1);
    errCublas = cublasGetError();
    if (errCublas != CUBLAS_STATUS_SUCCESS) goto stop;

    // Copy result to host.
    cudaMemcpy(vis, visd, nb * sizeof(float2), cudaMemcpyDeviceToHost);

    // Clean up before exit.
    stop:
    if (errCuda != cudaSuccess)
    {
        retVal = errCuda;
        printf("CUDA Error: %s\n", cudaGetErrorString(errCuda));
    }
    if (errCublas != CUBLAS_STATUS_SUCCESS)
    {
        retVal = errCublas;
        printf("CUBLAS Error: Code %d\n", errCublas);
    }

    // Free host memory.
    free(uvw);
    free(eb);
    free(n);

    // Free device memory.
    cudaFree(kmat);
    cudaFree(emat);
    cudaFree(uvwd);
    cudaFree(ld);
    cudaFree(md);
    cudaFree(nd);
    cudaFree(visd);

    // Shutdown.
    cublasShutdown();
    return retVal;
}

// Double precision.

int oskar_modules_cudad_correlator_lm_bw(int na, const double* ax,
        const double* ay, const double* az, int ns, const double* l,
        const double* m, const double* bsqrt, const double* e, double ra0,
        double dec0, double lst0, int nsdt, double sdt, double k,
        double bandwidth, double* vis, double* u, double* v, double* w)
{
    // Initialise.
    cudaError_t errCuda = cudaSuccess;
    cublasStatus errCublas = CUBLAS_STATUS_SUCCESS;
    double lst, ha0;
    double2 one = make_double2(1.0, 0.0);
    int i, a, retVal = 0;
    double tIncCentre, tInc;
    double tOffset = (double)sdt / 2.0;
    cublasInit();

    // Set up thread blocks.
    dim3 kThd(64, 4); // Sources, antennas.
    dim3 kBlk((ns + kThd.x - 1) / kThd.x, (na + kThd.y - 1) / kThd.y);
    size_t sMem = (kThd.x + kThd.y) * sizeof(double3);
    dim3 mThd(64, 4); // Sources, antennas.
    dim3 mBlk((ns + mThd.x - 1) / mThd.x, (na + mThd.y - 1) / mThd.y);
    dim3 vThd(256, 1); // Antennas, antennas.
    dim3 vBlk(na, na);
    size_t vsMem = vThd.x * sizeof(double2);

    // Get bandwidth term.
    double lambda_bandwidth = (2 * M_PI / k) * bandwidth;

    // Compute the source n-coordinates from l and m.
    double* n = (double*)malloc(ns * sizeof(double));
    double r = 0.0;
    for (i = 0; i < ns; ++i)
    {
        r = l[i]*l[i] + m[i]*m[i];
        n[i] = (r < 1.0) ? sqrt(1.0 - r) - 1.0 : -1.0;
    }

    // Scale source brightnesses (in Bsqrt) by station beams (in E).
    double2* eb = (double2*)malloc(ns * na * sizeof(double2));
    for (a = 0; a < na; ++a)
    {
        for (i = 0; i < ns; ++i)
        {
            int idx = i + a * ns;
            double bs = bsqrt[i];
            eb[idx].x = e[2*idx + 0] * bs; // Real
            eb[idx].y = e[2*idx + 1] * bs; // Imag
        }
    }

    // Allocate host memory for station u,v,w coordinates and visibility matrix.
    int nb = na * (na - 1) / 2;
    double* uvw = (double*)malloc(na * 3 * sizeof(double));

    // Allocate memory for source coordinates and visibility matrix on the
    // device.
    double *ld, *md, *nd, *uvwd;
    double2 *visd, *kmat, *emat;
    cudaMalloc((void**)&ld, ns * sizeof(double));
    cudaMalloc((void**)&md, ns * sizeof(double));
    cudaMalloc((void**)&nd, ns * sizeof(double));
    cudaMalloc((void**)&visd, nb * sizeof(double2));
    cudaMalloc((void**)&kmat, ns * na * sizeof(double2));
    cudaMalloc((void**)&emat, ns * na * sizeof(double2));
    cudaMalloc((void**)&uvwd, 3 * na * sizeof(double));
    cudaThreadSynchronize();
    errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) goto stop;

    // Copy source coordinates and modified E-matrix to device.
    cudaMemcpy(ld, l, ns * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(md, m, ns * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(nd, n, ns * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(emat, eb, ns * na * sizeof(double2), cudaMemcpyHostToDevice);

    // Clear visibility matrix.
    cudaMemset(visd, 0, nb * sizeof(double2));

    // Copy u,v,w baseline coordinates of mid-point to output arrays.
    // FIXME: probably don't need to return UVW from this function?
    tIncCentre = ((nsdt - 1) / 2) * sdt + tOffset;
    lst = lst0 + 2 * M_PI * tIncCentre * sdt / 86400.0f;
    ha0 = lst - ra0;
    oskar_math_synthesisd_xyz2uvw(na, ax, ay, az, ha0, dec0,
            &uvw[0], &uvw[na], &uvw[2*na]);
    oskar_math_synthesisd_baselines(na, &uvw[0], &uvw[na], &uvw[2*na],
            u, v, w);

    // Loop over integrations.
    for (i = 0; i < nsdt; ++i)
    {
        // Compute the current LST and hour angle of the phase centre.
        tInc = i * sdt + tOffset;
        lst = lst0 + 2 * M_PI * tInc / 86400.0;
        ha0 = lst - ra0;

        // Compute the station u,v,w coordinates.
        oskar_math_synthesisd_xyz2uvw(na, ax, ay, az, ha0, dec0,
                &uvw[0], &uvw[na], &uvw[2*na]);

        // Multiply station u,v,w coordinates by 2 pi / lambda.
        for (a = 0; a < na; ++a)
        {
            uvw[a] *= k;
            uvw[na + a] *= k;
            uvw[2 * na + a] *= k;
        }

        // Copy u,v,w coordinates to device.
        cudaMemcpy(uvwd, uvw, na * 3 * sizeof(double), cudaMemcpyHostToDevice);

        // Compute K-matrix.
        oskar_cudakd_dftw_3d_seq_out <<<kBlk, kThd, sMem>>> (
                na, &uvwd[0], &uvwd[na], &uvwd[2*na], ns, ld, md, nd, kmat);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) goto stop;

        // Perform complex matrix element multiply.
        oskar_cudakd_mat_mul_cc <<<mBlk, mThd>>> (ns, na, kmat, emat, kmat);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) goto stop;

        // Call the correlator kernel.
        oskar_cudakd_correlator <<<vBlk, vThd, vsMem>>> (
                ns, na, kmat, &uvwd[0], &uvwd[na], ld, md, lambda_bandwidth, visd);
        cudaThreadSynchronize();
        errCuda = cudaPeekAtLastError();
        if (errCuda != cudaSuccess) goto stop;
    }

    // Scale result.
    cublasZscal(nb, make_double2(1.0 / nsdt, 0.0), visd, 1);
    errCublas = cublasGetError();
    if (errCublas != CUBLAS_STATUS_SUCCESS) goto stop;

    // Copy result to host.
    cudaMemcpy(vis, visd, nb * sizeof(double2), cudaMemcpyDeviceToHost);

    // Clean up before exit.
    stop:
    if (errCuda != cudaSuccess)
    {
        retVal = errCuda;
        printf("CUDA Error: %s\n", cudaGetErrorString(errCuda));
    }
    if (errCublas != CUBLAS_STATUS_SUCCESS)
    {
        retVal = errCublas;
        printf("CUBLAS Error: Code %d\n", errCublas);
    }

    // Free host memory.
    free(uvw);
    free(eb);
    free(n);

    // Free device memory.
    cudaFree(kmat);
    cudaFree(emat);
    cudaFree(uvwd);
    cudaFree(ld);
    cudaFree(md);
    cudaFree(nd);
    cudaFree(visd);

    // Shutdown.
    cublasShutdown();
    return retVal;
}

#ifdef __cplusplus
}
#endif
