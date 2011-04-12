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
#include <stdio.h>
#include <cublas.h>

#include "cuda/CudaEclipse.h"

#ifdef __cplusplus
extern "C" {
#endif

//__constant__ float uvw[];

void oskar_modules_cuda_correlator_lm(int na, const float* ax, const float* ay,
        const float* az, int ns, const float* l, const float* m,
        const float* bsqrt, const float* e, float ra0, float dec0,
        float lst0, int nsdt, float sdt, float k, float* vis, float* u,
        float* v, float* w)
{
//    // Allocate memory for antenna positions and source coordinates
//    // on the device.
//    float *axd, *ayd, *azd, *ld, *md, *nd;
//    float2 *weightsd;
//    cudaMalloc((void**)&axd, na * sizeof(float));
//    cudaMalloc((void**)&ayd, na * sizeof(float));
//    cudaMalloc((void**)&azd, na * sizeof(float));
//    cudaMalloc((void**)&ld, ns * sizeof(float));
//    cudaMalloc((void**)&md, ns * sizeof(float));
//    cudaMalloc((void**)&nd, ns * sizeof(float));
//    cudaMalloc((void**)&weightsd, ns * na * sizeof(float2));
//
//    // Copy antenna positions and source coordinates to device.
//    cudaMemcpy(axd, ax, na * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(ayd, ay, na * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(azd, az, na * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(ld, l, ns * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(md, m, ns * sizeof(float), cudaMemcpyHostToDevice);
//    cudaMemcpy(nd, n, ns * sizeof(float), cudaMemcpyHostToDevice);



//    // Free device memory.
//    cudaFree(axd);
//    cudaFree(ayd);
//    cudaFree(azd);
//    cudaFree(ld);
//    cudaFree(md);
//    cudaFree(nd);
//    cudaFree(weightsd);
}

#ifdef __cplusplus
}
#endif
