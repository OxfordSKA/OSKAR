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

#include "cuda/kernels/oskar_cudak_crvecmul.h"

// Single precision.

__global__
void oskar_cudakf_crvecmul(int n, const float2* a, const float* b, float2* c)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        // Cache the input data.
        float2 ac = a[i];
        float bc = b[i];

        // Multiply.
        float2 cc;
        cc.x = ac.x * bc;
        cc.y = ac.y * bc;

        c[i] = cc;
    }
}

// Double precision.

__global__
void oskar_cudakd_crvecmul(int n, const double2* a, const double* b,
        double2* c)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n)
    {
        // Cache the input data.
        double2 ac = a[i];
        double bc = b[i];

        // Multiply.
        double2 cc;
        cc.x = ac.x * bc;
        cc.y = ac.y * bc;

        c[i] = cc;
    }
}
