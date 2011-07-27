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

#include "sky/cudak/oskar_sky_cudak_hor_lmn_to_phi_theta.h"

// Single precision.

__global__
void oskar_sky_cudakf_hor_lmn_to_phi_theta(int n, const float* p_l,
        const float* p_m, const float* p_n, float* phi, float* theta)
{
    // Get the position ID that this thread is working on.
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    // Get the data.
    float x = p_l[i];
    float y = p_m[i];
    float z = p_n[i];
    __syncthreads();

    // Cartesian to spherical.
    float p = atan2f(y, x); // Phi.
    x = sqrtf(x*x + y*y);
    y = atan2f(x, z); // Theta.
    phi[i] = p;
    theta[i] = y;
}

// Double precision.

__global__
void oskar_sky_cudakd_hor_lmn_to_phi_theta(int n, const double* p_l,
        const double* p_m, const double* p_n, double* phi, double* theta)
{
    // Get the position ID that this thread is working on.
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    // Get the data.
    double x = p_l[i];
    double y = p_m[i];
    double z = p_n[i];
    __syncthreads();

    // Cartesian to spherical.
    double p = atan2(y, x); // Phi.
    x = sqrt(x*x + y*y);
    y = atan2(x, z); // Theta.
    phi[i] = p;
    theta[i] = y;
}
