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

#include "cuda/oskar_cuda_bfmv.h"
#include <cublas.h>

#include "cuda/CudaEclipse.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_cuda_bfmv(const unsigned na, const unsigned nb,
        const float* signals, const float* weights, float* beams)
{
    // Initialise cuBLAS.
    cublasInit();

    // Allocate memory for antenna signals and beamforming weights
    // on the device.
    float2 *signalsd, *weightsd, *beamsd;
    cudaMalloc((void**)&signalsd, na * sizeof(float2));
    cudaMalloc((void**)&beamsd, nb * sizeof(float2));
    cudaMalloc((void**)&weightsd, na * nb * sizeof(float2));

    // Copy antenna signals and beamforming weights to the device.
    cudaMemcpy(signalsd, signals, na * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(weightsd, weights, na * nb * sizeof(float2), cudaMemcpyHostToDevice);

    // Call cuBLAS function to perform the matrix-vector multiplication.
    // Note that cuBLAS calls use Fortran-ordering (column major) for their
    // matrices, so we use the transpose here.
    cublasCgemv('t', na, nb, make_float2(1.0, 0.0),
            weightsd, na, signalsd, 1, make_float2(0.0, 0.0), beamsd, 1);

    // Copy result from device memory to host memory.
    cudaMemcpy(beams, beamsd, nb * sizeof(float2), cudaMemcpyDeviceToHost);

    // Free device memory.
    cudaFree(signalsd);
    cudaFree(weightsd);
    cudaFree(beamsd);

    // Shut down cuBLAS.
    cublasShutdown();
}

#ifdef __cplusplus
}
#endif
