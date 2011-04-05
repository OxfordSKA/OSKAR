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

#include "cuda/oskar_cudad_bp2hssgu.h"
#include "cuda/kernels/oskar_cudakd_bp2hssw.h"
#include "cuda/kernels/oskar_cudakd_wt2hgu.h"
#include <stdio.h>

#include "cuda/CudaEclipse.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_cudad_bp2hssgu(int na, const double* ax, const double* ay,
        int ns, const double* slon, const double* slat,
        double ba, double be, double k, double* image)
{
    // Precompute.
    double3 trig = make_double3(cos(ba), sin(ba), cos(be));

    // Allocate memory for antenna positions, antenna weights,
    // test source positions and pixel values on the device.
    double *axd, *ayd, *slond, *slatd;
    double2 *weights, *pix;
    double3 *trigd;
    cudaMalloc((void**)&axd, na * sizeof(double));
    cudaMalloc((void**)&ayd, na * sizeof(double));
    cudaMalloc((void**)&weights, na * sizeof(double2));
    cudaMalloc((void**)&trigd, 1 * sizeof(double3));

    // Copy antenna positions and beam geometry to device.
    cudaMemcpy(axd, ax, na * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(ayd, ay, na * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(trigd, &trig, 1 * sizeof(double3), cudaMemcpyHostToDevice);

    // Invoke kernel to compute unnormalised antenna weights on the device.
    dim3 wThreads(256, 1);
    dim3 wBlocks((na + wThreads.x - 1) / wThreads.x, 1);
    size_t wSharedMem = wThreads.x * sizeof(double2) + sizeof(double3);
    oskar_cudakd_wt2hgu <<<wBlocks, wThreads, wSharedMem>>> (
            na, axd, ayd, 1, trigd, k, weights);
    cudaThreadSynchronize();

    // Divide up the source (pixel) list into manageable chunks.
    int nsMax = 100000;
    int chunk = 0, chunks = (ns + nsMax - 1) / nsMax;

    // Allocate memory for source position chunk on the device.
    cudaMalloc((void**)&slond, nsMax * sizeof(double));
    cudaMalloc((void**)&slatd, nsMax * sizeof(double));
    cudaMalloc((void**)&pix, nsMax * sizeof(double2));

    // Loop over pixel chunks.
    for (chunk = 0; chunk < chunks; ++chunk) {
        const int srcStart = chunk * nsMax;
        int srcInBlock = ns - srcStart;
        if (srcInBlock > nsMax) srcInBlock = nsMax;

        // Copy test source positions for this chunk to the device.
        cudaMemcpy(slond, slon + srcStart, srcInBlock * sizeof(double),
                cudaMemcpyHostToDevice);
        cudaMemcpy(slatd, slat + srcStart, srcInBlock * sizeof(double),
                cudaMemcpyHostToDevice);

        // Invoke kernel to compute the (partial) beam pattern on the device.
        int threadsPerBlock = 256;
        int blocks = (srcInBlock + threadsPerBlock - 1) / threadsPerBlock;
        int maxAntennasPerBlock = 864; // Should be multiple of 16.
        size_t sharedMem = 2 * maxAntennasPerBlock * sizeof(double2);
        oskar_cudakd_bp2hssw <<<blocks, threadsPerBlock, sharedMem>>>
                (na, axd, ayd, weights, srcInBlock, slond, slatd, k,
                        maxAntennasPerBlock, pix);
        cudaThreadSynchronize();
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess)
            printf("CUDA Error: %s\n", cudaGetErrorString(err));

        // Copy (partial) result from device memory to host memory.
        cudaMemcpy(image + 2 * srcStart, pix, srcInBlock * sizeof(double2),
                cudaMemcpyDeviceToHost);
    }

    // Free device memory.
    cudaFree(axd);
    cudaFree(ayd);
    cudaFree(weights);
    cudaFree(slond);
    cudaFree(slatd);
    cudaFree(pix);
    cudaFree(trigd);
}

#ifdef __cplusplus
}
#endif
