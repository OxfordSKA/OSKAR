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

#include "cuda/oskar_cuda_im2dft.h"
#include "cuda/kernels/oskar_cudak_im2dft.h"
#include <stdio.h>
#include <math.h>

#include "cuda/CudaEclipse.h"


#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.1415926535f
#endif

void oskar_cuda_im2dft(int nv, const float* u, const float* v,
        const float* vis, int nl, int nm, float dl, float dm,
        float sl, float sm, float* image)
{
    // Get the centre pixel in L and M.
    const int centreL = (int)floor(nl / 2.0f);
    const int centreM = (int)floor(nm / 2.0f);

    // Create and allocate memory for the pixel positions.
    const int np = nl * nm; // Number of pixels in image.
    float* pl = (float*)malloc(np * sizeof(float));
    float* pm = (float*)malloc(np * sizeof(float));
    int i, j, k; // Indices.
    float l, m; // Pixel coordinates.
    for (j = 0; j < nm; ++j) {
        // Image m-coordinate.
        m = 2.0f * (j - centreM) * dm * sm / M_PI;

        for (i = 0; i < nl; ++i) {
            // Image l-coordinate.
            l = 2.0f * (i - centreL) * dl * sl / M_PI;

            // Image pixel index.
            k = i + j * nl;
            pl[k] = l;
            pm[k] = m;
        }
    }

    // Allocate memory for visibilities and u,v-coordinates on the device.
    float *ud, *vd, *pld, *pmd, *pix;
    float2 *visd;
    cudaMalloc((void**)&ud, nv * sizeof(float));
    cudaMalloc((void**)&vd, nv * sizeof(float));
    cudaMalloc((void**)&visd, nv * sizeof(float2));

    // Copy visibilities and u,v-coordinates to device.
    cudaMemcpy(ud, u, nv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(vd, v, nv * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(visd, vis, nv * sizeof(float2), cudaMemcpyHostToDevice);

    // Divide up the pixel list into manageable chunks.
    int npMax = 100000;
    int chunk = 0, chunks = (np + npMax - 1) / npMax;

    // Allocate memory for pixel position chunk on the device.
    cudaMalloc((void**)&pld, npMax * sizeof(float));
    cudaMalloc((void**)&pmd, npMax * sizeof(float));
    cudaMalloc((void**)&pix, npMax * sizeof(float));

    // Loop over pixel chunks.
    for (chunk = 0; chunk < chunks; ++chunk) {
        const int pixStart = chunk * npMax;
        int pixInBlock = np - pixStart;
        if (pixInBlock > npMax) pixInBlock = npMax;

        // Copy test source positions for this chunk to the device.
        cudaMemcpy(pld, pl + pixStart, pixInBlock * sizeof(float),
                cudaMemcpyHostToDevice);
        cudaMemcpy(pmd, pm + pixStart, pixInBlock * sizeof(float),
                cudaMemcpyHostToDevice);

        // Invoke kernel to compute the (partial) image on the device.
        int threadsPerBlock = 384;
        int blocks = (pixInBlock + threadsPerBlock - 1) / threadsPerBlock;
        int maxVisPerBlock = 896; // Should be multiple of 16.
        size_t sharedMem = 2 * maxVisPerBlock * sizeof(float2);
        oskar_cudak_im2dft <<<blocks, threadsPerBlock, sharedMem>>>
        		(nv, ud, vd, visd, pixInBlock, pld, pmd, maxVisPerBlock, pix);
        cudaThreadSynchronize();
        cudaError_t err = cudaPeekAtLastError();
        if (err != cudaSuccess)
            printf("CUDA Error: %s\n", cudaGetErrorString(err));

        // Copy (partial) result from device memory to host memory.
        cudaMemcpy(image + pixStart, pix, pixInBlock * sizeof(float),
                cudaMemcpyDeviceToHost);
    }

    // Free device memory.
    cudaFree(ud);
    cudaFree(vd);
    cudaFree(visd);
    cudaFree(pld);
    cudaFree(pmd);
    cudaFree(pix);

    // Free host memory.
    free(pm);
    free(pl);
}

#ifdef __cplusplus
}
#endif
