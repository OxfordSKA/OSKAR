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

#include "utility/oskar_cuda_memory.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_cuda_malloc(void** ptr, unsigned size)
{
    cudaMalloc(ptr, size);
}

void oskar_cuda_malloc_double(double** ptr, unsigned n)
{
    cudaMalloc((void**)ptr, n * sizeof(double));
}

void oskar_cuda_malloc_float(float** ptr, unsigned n)
{
    cudaMalloc((void**)ptr, n * sizeof(float));
}

void oskar_cuda_malloc_int(int** ptr, unsigned n)
{
    cudaMalloc((void**)ptr, n * sizeof(int));
}

void oskar_cuda_memcpy_h2d(void* dest, const void* src, unsigned size)
{
    cudaMemcpy(dest, src, size, cudaMemcpyHostToDevice);
}

void oskar_cuda_memcpy_h2d_double(double* dest, const double* src, unsigned n)
{
    cudaMemcpy((void*)dest, (const void*)src, n * sizeof(double),
            cudaMemcpyHostToDevice);
}

void oskar_cuda_memcpy_h2d_float(float* dest, const float* src, unsigned n)
{
    cudaMemcpy((void*)dest, (const void*)src, n * sizeof(float),
            cudaMemcpyHostToDevice);
}

void oskar_cuda_memcpy_h2d_int(int* dest, const int* src, unsigned n)
{
    cudaMemcpy((void*)dest, (const void*)src, n * sizeof(int),
            cudaMemcpyHostToDevice);
}

void oskar_cuda_memcpy_d2h(void* dest, const void* src, unsigned size)
{
    cudaMemcpy(dest, src, size, cudaMemcpyDeviceToHost);
}

void oskar_cuda_memcpy_d2h_double(double* dest, const double* src, unsigned n)
{
    cudaMemcpy((void*)dest, (const void*)src, n * sizeof(double),
            cudaMemcpyDeviceToHost);
}

void oskar_cuda_memcpy_d2h_float(float* dest, const float* src, unsigned n)
{
    cudaMemcpy((void*)dest, (const void*)src, n * sizeof(float),
            cudaMemcpyDeviceToHost);
}

void oskar_cuda_memcpy_d2h_int(int* dest, const int* src, unsigned n)
{
    cudaMemcpy((void*)dest, (const void*)src, n * sizeof(int),
            cudaMemcpyDeviceToHost);
}

void oskar_cuda_free(void* ptr)
{
    cudaFree(ptr);
}

#ifdef __cplusplus
}
#endif
