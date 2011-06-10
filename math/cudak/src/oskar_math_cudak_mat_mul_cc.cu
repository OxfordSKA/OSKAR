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

#include "cuda/kernels/oskar_cudak_mat_mul_cc.h"

// Single precision.

__global__
void oskar_cudakf_mat_mul_cc(int n1, int n2, const float2* a, const float2* b,
        float2* c)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x; // Fastest varying.
    int j = blockDim.y * blockIdx.y + threadIdx.y; // Slowest varying.
    if (i < n1 && j < n2)
    {
        // Compute matrix index.
        int idx = i + j * n1;

        // Cache the input data.
        float2 ac = a[idx];
        float2 bc = b[idx];

        // Complex multiply.
        float2 cc;
        cc.x = ac.x * bc.x - ac.y * bc.y; // RE*RE - IM*IM
        cc.y = ac.y * bc.x + ac.x * bc.y; // IM*RE + RE*IM
        c[idx] = cc;
    }
}

// Double precision.

__global__
void oskar_cudakd_mat_mul_cc(int n1, int n2, const double2* a, const double2* b,
        double2* c)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x; // Fastest varying.
    int j = blockDim.y * blockIdx.y + threadIdx.y; // Slowest varying.
    if (i < n1 && j < n2)
    {
        // Compute matrix index.
        int idx = i + j * n1;

        // Cache the input data.
        double2 ac = a[idx];
        double2 bc = b[idx];

        // Complex multiply.
        double2 cc;
        cc.x = ac.x * bc.x - ac.y * bc.y; // RE*RE - IM*IM
        cc.y = ac.y * bc.x + ac.x * bc.y; // IM*RE + RE*IM
        c[idx] = cc;
    }
}
