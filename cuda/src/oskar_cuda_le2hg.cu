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

#include "cuda/oskar_cuda_le2hg.h"
#include "cuda/kernels/oskar_cudak_le2hg.h"
#include <stdio.h>

#include "cuda/CudaEclipse.h"

#define TIMER_ENABLE 1
#include "utility/timer.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.

void oskar_cudaf_le2hg(char opt, int ns, const float* hadec, const float* dec,
        float cosLat, float sinLat, float* azel, float* el)
{
    // Device pointers.
    float2 *hadecd, *azeld;
    if (opt == 'D' || opt == 'd') {
        hadecd = (float2*)hadec;
        azeld = (float2*)azel;
    }
    else {
        // Allocate device memory.
        cudaMalloc((void**)&hadecd, ns * sizeof(float2));
        cudaMalloc((void**)&azeld,  ns * sizeof(float2));
    }

    // Host pointers.
    float *hadech, *azelh;
    if (opt == 'S' || opt == 's') {
        // Allocate host memory.
        hadech = (float*)malloc(2 * ns * sizeof(float));
        azelh  = (float*)malloc(2 * ns * sizeof(float));

        // Interleave coordinates.
        for (int i = 0; i < ns; ++i) {
            hadech[2 * i + 0] = hadec[i];
            hadech[2 * i + 1] = dec[i];
        }
    }
    else {
        hadech = (float*)hadec;
        azelh = (float*)azel;
    }

    // Copy host memory to device if required.
    if (!(opt == 'D' || opt == 'd')) {
        cudaMemcpy(hadecd, hadech, ns * sizeof(float2),
                cudaMemcpyHostToDevice);
    }

    // Invoke kernel to compute horizontal coordinates on the device.
    int threadsPerBlock = 512;
    int blocks = (ns + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMem = threadsPerBlock * sizeof(float2);
    oskar_cudakf_le2hg <<<blocks, threadsPerBlock, sharedMem>>>
            (ns, hadecd, cosLat, sinLat, azeld);
    cudaThreadSynchronize();
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Copy device memory to host if required.
    if (!(opt == 'D' || opt == 'd')) {
        cudaMemcpy(azelh, azeld, ns * sizeof(float2),
                cudaMemcpyDeviceToHost);
    }

    // Separate coordinates if required.
    if (opt == 'S' || opt == 's') {
        for (int i = 0; i < ns; ++i) {
            azel[i] = azelh[2 * i + 0];
            el[i]   = azelh[2 * i + 1];
        }
    }

    // Free memory if allocated.
    if (!(opt == 'D' || opt == 'd')) {
        cudaFree(hadecd);
        cudaFree(azeld);
    }
    if (opt == 'S' || opt == 's') {
        free(hadech);
        free(azelh);
    }
}

// Double precision.

void oskar_cudad_le2hg(char opt, int ns, const double* hadec, const double* dec,
        double cosLat, double sinLat, double* azel, double* el)
{
    // Device pointers.
    double2 *hadecd, *azeld;
    if (opt == 'D' || opt == 'd') {
        hadecd = (double2*)hadec;
        azeld = (double2*)azel;
    }
    else {
        // Allocate device memory.
        cudaMalloc((void**)&hadecd, ns * sizeof(double2));
        cudaMalloc((void**)&azeld,  ns * sizeof(double2));
    }

    // Host pointers.
    double *hadech, *azelh;
    if (opt == 'S' || opt == 's') {
        // Allocate host memory.
        hadech = (double*)malloc(2 * ns * sizeof(double));
        azelh  = (double*)malloc(2 * ns * sizeof(double));

        // Interleave coordinates.
        for (int i = 0; i < ns; ++i) {
            hadech[2 * i + 0] = hadec[i];
            hadech[2 * i + 1] = dec[i];
        }
    }
    else {
        hadech = (double*)hadec;
        azelh = (double*)azel;
    }

    // Copy host memory to device if required.
    if (!(opt == 'D' || opt == 'd')) {
        cudaMemcpy(hadecd, hadech, ns * sizeof(double2),
                cudaMemcpyHostToDevice);
    }

    // Invoke kernel to compute horizontal coordinates on the device.
    int threadsPerBlock = 256;
    int blocks = (ns + threadsPerBlock - 1) / threadsPerBlock;
    size_t sharedMem = threadsPerBlock * sizeof(double2);
    oskar_cudakd_le2hg <<<blocks, threadsPerBlock, sharedMem>>>
            (ns, hadecd, cosLat, sinLat, azeld);
    cudaThreadSynchronize();
    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess)
        printf("CUDA Error: %s\n", cudaGetErrorString(err));

    // Copy device memory to host if required.
    if (!(opt == 'D' || opt == 'd')) {
        cudaMemcpy(azelh, azeld, ns * sizeof(double2),
                cudaMemcpyDeviceToHost);
    }

    // Separate coordinates if required.
    if (opt == 'S' || opt == 's') {
        for (int i = 0; i < ns; ++i) {
            azel[i] = azelh[2 * i + 0];
            el[i]   = azelh[2 * i + 1];
        }
    }

    // Free memory if allocated.
    if (!(opt == 'D' || opt == 'd')) {
        cudaFree(hadecd);
        cudaFree(azeld);
    }
    if (opt == 'S' || opt == 's') {
        free(hadech);
        free(azelh);
    }
}

#ifdef __cplusplus
}
#endif
