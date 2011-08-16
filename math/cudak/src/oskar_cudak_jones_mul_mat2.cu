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

#include "math/cudak/oskar_cudak_jones_mul_mat2.h"
#include "math/cudak/oskar_cudaf_mul_mat2c_mat2c.h"

// Single precision.

__global__
void oskar_cudak_jones_mul_mat2_f(int n, const float4c* j1,
        const float4c* j2, float4c* m)
{
    // Get the array index ID that this thread is working on.
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Get the data from global memory.
    float4c c_j1, c_j2, c_m;
    if (i < n)
    {
        c_j1 = j1[i];
        c_j2 = j2[i];
    }
    __syncthreads();

    // Multiply the two complex matrices.
    oskar_cudaf_mul_mat2c_mat2c_f(c_j1, c_j2, c_m);

    // Copy result back to global memory.
    __syncthreads();
    if (i < n)
        m[i] = c_m;
}

// Double precision.

__global__
void oskar_cudak_jones_mul_mat2_d(int n, const double4c* j1,
        const double4c* j2, double4c* m)
{
    // Get the array index ID that this thread is working on.
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Get the data from global memory.
    double4c c_j1, c_j2, c_m;
    if (i < n)
    {
        c_j1 = j1[i];
        c_j2 = j2[i];
    }
    __syncthreads();

    // Multiply the two complex matrices.
    oskar_cudaf_mul_mat2c_mat2c_d(c_j1, c_j2, c_m);

    // Copy result back to global memory.
    __syncthreads();
    if (i < n)
        m[i] = c_m;
}
