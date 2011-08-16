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

#include "math/cudak/oskar_cudak_jones_mul_c2.h"
#include "math/cudak/oskar_cudaf_mul_c_c.h"

// Single precision.

__global__
void oskar_cudak_jones_mul_c2_f(int n, const float2* s1,
        const float2* s2, float4c* m)
{
    // Get the array index ID that this thread is working on.
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Get the data from global memory.
    float2 c_s1, c_s2;
    float4c c_m;
    if (i < n)
    {
        c_s1 = s1[i];
        c_s2 = s2[i];
    }
    __syncthreads();

    // Multiply the two complex numbers and store them in a Jones matrix.
    float2 t;
    oskar_cudaf_mul_c_c_f(c_s1, c_s2, t);
    c_m.a = t;
    c_m.b = make_float2(0.0f, 0.0f);
    c_m.c = make_float2(0.0f, 0.0f);
    c_m.d = t;

    // Copy result back to global memory.
    __syncthreads();
    if (i < n)
        m[i] = c_m;
}

// Double precision.

__global__
void oskar_cudak_jones_mul_c2_d(int n, const double2* s1,
        const double2* s2, double4c* m)
{
    // Get the array index ID that this thread is working on.
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Get the data from global memory.
    double2 c_s1, c_s2;
    double4c c_m;
    if (i < n)
    {
        c_s1 = s1[i];
        c_s2 = s2[i];
    }
    __syncthreads();

    // Multiply the two complex numbers and store them in a Jones matrix.
    double2 t;
    oskar_cudaf_mul_c_c_d(c_s1, c_s2, t);
    c_m.a = t;
    c_m.b = make_double2(0.0, 0.0);
    c_m.c = make_double2(0.0, 0.0);
    c_m.d = t;

    // Copy result back to global memory.
    __syncthreads();
    if (i < n)
        m[i] = c_m;
}
