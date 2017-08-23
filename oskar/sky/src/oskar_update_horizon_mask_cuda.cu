/*
 * Copyright (c) 2011-2017, The University of Oxford
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

#include "sky/oskar_update_horizon_mask_cuda.h"
#include <math.h>

template<typename T>
__global__
void oskar_update_horizon_mask_cudak(const int num_sources,
        const T* restrict l, const T* restrict m, const T* restrict n,
        const T l_mul, const T m_mul, const T n_mul, int* restrict mask)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_sources) return;
    mask[i] |= ((l[i] * l_mul + m[i] * m_mul + n[i] * n_mul) > (T) 0.);
}

void oskar_update_horizon_mask_cuda_f(int num_sources, const float* d_l,
        const float* d_m, const float* d_n, const float l_mul,
        const float m_mul, const float n_mul, int* d_mask)
{
    int num_threads = 256;
    int num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_update_horizon_mask_cudak<float>
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (
            num_sources, d_l, d_m, d_n, l_mul, m_mul, n_mul, d_mask);
}

void oskar_update_horizon_mask_cuda_d(int num_sources, const double* d_l,
        const double* d_m, const double* d_n, const double l_mul,
        const double m_mul, const double n_mul, int* d_mask)
{
    int num_threads = 256;
    int num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_update_horizon_mask_cudak<double>
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (
            num_sources, d_l, d_m, d_n, l_mul, m_mul, n_mul, d_mask);
}
