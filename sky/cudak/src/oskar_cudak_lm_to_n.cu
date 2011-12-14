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

#include "sky/cudak/oskar_cudak_lm_to_n.h"

// Single precision.

__global__
void oskar_cudak_lm_to_n_f(int n, const float* p_l, const float* p_m,
        float* p_n)
{
    // Get the position ID that this thread is working on.
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    float l = p_l[i];
    float m = p_m[i];
    float a = 1.0f - l*l - m*m;
    if (a < 0.0f)
    {
        p_n[i] = -1.0f;
    }
    else
    {
        float x = sqrtf(a) - 1.0f;
        p_n[i] = x;
    }
}

// Double precision.

__global__
void oskar_cudak_lm_to_n_d(int n, const double* p_l, const double* p_m,
        double* p_n)
{
    // Get the position ID that this thread is working on.
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    double l = p_l[i];
    double m = p_m[i];
    double a = 1.0 - l*l - m*m;
    if (a < 0.0)
    {
        p_n[i] = -1.0;
    }
    else
    {
        double x = sqrt(a) - 1.0;
        p_n[i] = x;
    }
}
