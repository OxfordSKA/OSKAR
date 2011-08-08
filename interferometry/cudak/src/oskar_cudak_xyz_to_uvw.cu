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

#include "interferometry/cudak/oskar_cudak_xyz_to_uvw.h"

// Single precision.
__global__
void oskar_cudak_xyz_to_uvw_f(int n, const float* x, const float* y,
        const float* z, float ha0, float dec0, float* u, float* v, float* w)
{
    // Pre-compute sine and cosine of input angles.
    __device__ __shared__ float sinHa0, cosHa0, sinDec0, cosDec0;
    if (threadIdx.x == 0)
    {
        sincosf(ha0, &sinHa0, &cosHa0);
        sincosf(dec0, &sinDec0, &cosDec0);
    }
    __syncthreads();

    // Get station ID.
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    // Cache input coordinates.
    float cx = x[i], cy = y[i], cz = z[i], cv, cw, t;

    // Do the rotation.
    t = cx * cosHa0;
    t -= cy * sinHa0;
    cv = cz * cosDec0;
    cv -= sinDec0 * t;
    cw = cosDec0 * t;
    cw += cz * sinDec0;
    t =  cx * sinHa0;
    t += cy * cosHa0;
    u[i] = t;
    v[i] = cv;
    w[i] = cw;
}

// Double precision.
__global__
void oskar_cudak_xyz_to_uvw_d(int n, const double* x, const double* y,
        const double* z, double ha0, double dec0, double* u, double* v,
        double* w)
{
    // Pre-compute sine and cosine of input angles.
    __device__ __shared__ double sinHa0, cosHa0, sinDec0, cosDec0;
    if (threadIdx.x == 0)
    {
        sincos(ha0, &sinHa0, &cosHa0);
        sincos(dec0, &sinDec0, &cosDec0);
    }
    __syncthreads();

    // Get station ID.
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    // Cache input coordinates.
    double cx = x[i], cy = y[i], cz = z[i], cv, cw, t;

    // Do the rotation.
    t = cx * cosHa0;
    t -= cy * sinHa0;
    cv = cz * cosDec0;
    cv -= sinDec0 * t;
    cw = cosDec0 * t;
    cw += cz * sinDec0;
    t =  cx * sinHa0;
    t += cy * cosHa0;
    u[i] = t;
    v[i] = cv;
    w[i] = cw;
}
