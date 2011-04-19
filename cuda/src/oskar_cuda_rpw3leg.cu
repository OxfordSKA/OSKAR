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
#include "cuda/kernels/oskar_cudak_rpw3leg.h"
#include <stdio.h>

#include "cuda/CudaEclipse.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.

void oskar_cudaf_rpw3leg(int na, float* ax, float* ay, float* az, int ns,
        float* ha, float* dec, float ha0, float dec0, float k, float* weights)
{
    // Allocate memory for antenna positions and source coordinates
    // on the device.
    float *axd, *ayd, *azd, *had, *decd;
    float2 *weightsd;
    cudaMalloc((void**)&axd, na * sizeof(float));
    cudaMalloc((void**)&ayd, na * sizeof(float));
    cudaMalloc((void**)&azd, na * sizeof(float));
    cudaMalloc((void**)&had, ns * sizeof(float));
    cudaMalloc((void**)&decd, ns * sizeof(float));
    cudaMalloc((void**)&weightsd, ns * na * sizeof(float2));

    // Copy antenna positions and source coordinates to device.
    cudaMemcpy(axd, ax, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ayd, ay, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(azd, az, na * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(had, ha, ns * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(decd, dec, ns * sizeof(float), cudaMemcpyHostToDevice);

    // Precompute.
    float2 scha0 = make_float2(sin(ha0), cos(ha0));
    float2 scdec0 = make_float2(sin(dec0), cos(dec0));

    // Invoke kernel to compute phase weights on the device.
    dim3 threads(64, 4); // Sources, antennas.
    dim3 blocks((ns + threads.x - 1) / threads.x,
            (na + threads.y - 1) / threads.y);
    size_t sharedMem = threads.x * sizeof(float2)
                        + threads.y * sizeof(float3);
    oskar_cudakf_rpw3leg <<<blocks, threads, sharedMem>>> (
            na, axd, ayd, azd, scha0, scdec0, ns, had, decd, k, weightsd);
    cudaThreadSynchronize();
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Copy result from device memory to host memory.
    cudaMemcpy(weights, weightsd, ns * na * sizeof(float2),
            cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(axd);
    cudaFree(ayd);
    cudaFree(azd);
    cudaFree(had);
    cudaFree(decd);
    cudaFree(weightsd);
}

// Double precision.

void oskar_cudad_rpw3leg(int na, double* ax, double* ay, double* az, int ns,
        double* ha, double* dec, double ha0, double dec0, double k, double* weights)
{
    // Allocate memory for antenna positions and source coordinates
    // on the device.
    double *axd, *ayd, *azd, *had, *decd;
    double2 *weightsd;
    cudaMalloc((void**)&axd, na * sizeof(double));
    cudaMalloc((void**)&ayd, na * sizeof(double));
    cudaMalloc((void**)&azd, na * sizeof(double));
    cudaMalloc((void**)&had, ns * sizeof(double));
    cudaMalloc((void**)&decd, ns * sizeof(double));
    cudaMalloc((void**)&weightsd, ns * na * sizeof(double2));

    // Copy antenna positions and source coordinates to device.
    cudaMemcpy(axd, ax, na * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ayd, ay, na * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(azd, az, na * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(had, ha, ns * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(decd, dec, ns * sizeof(double), cudaMemcpyHostToDevice);

    // Precompute.
    double2 scha0 = make_double2(sin(ha0), cos(ha0));
    double2 scdec0 = make_double2(sin(dec0), cos(dec0));

    // Invoke kernel to compute phase weights on the device.
    dim3 threads(64, 4); // Sources, antennas.
    dim3 blocks((ns + threads.x - 1) / threads.x,
            (na + threads.y - 1) / threads.y);
    size_t sharedMem = threads.x * sizeof(double2)
                        + threads.y * sizeof(double3);
    oskar_cudakd_rpw3leg <<<blocks, threads, sharedMem>>> (
            na, axd, ayd, azd, scha0, scdec0, ns, had, decd, k, weightsd);
    cudaThreadSynchronize();
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Copy result from device memory to host memory.
    cudaMemcpy(weights, weightsd, ns * na * sizeof(double2),
            cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(axd);
    cudaFree(ayd);
    cudaFree(azd);
    cudaFree(had);
    cudaFree(decd);
    cudaFree(weightsd);
}


#ifdef __cplusplus
}
#endif
