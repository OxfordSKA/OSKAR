/*
 * Copyright (c) 2013, The University of Oxford
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

#include <oskar_convert_apparent_ha_dec_to_enu_direction_cosines_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

// Single precision
__global__
void oskar_convert_apparent_ha_dec_to_enu_direction_cosines_cudak_f(int n,
        const float* ha, const float* dec, float cosLat, float sinLat, float* x,
        float* y, float* z)
{
    // Get the coordinate index that this thread is working on.
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Copy local equatorial coordinates from global memory.
    float sh, sd; // Source HA, Dec.
    if (idx < n)
    {
        sh = ha[idx];
        sd = dec[idx];
    }
    __syncthreads(); // Coalesce memory accesses.

    // Find direction cosines.
    float cosDec, sinDec, cosHA, sinHA, t, X1, Y2;
    sincosf(sh, &sinHA, &cosHA);
    sincosf(sd, &sinDec, &cosDec);
    t = cosDec * cosHA;
    X1 = cosLat * sinDec - sinLat * t;
    Y2 = sinLat * sinDec + cosLat * t;
    t = -cosDec * sinHA;

    // Copy direction cosines into global memory.
    __syncthreads(); // Coalesce memory accesses.
    if (idx < n)
    {
        x[idx] = t;  // Horizontal x-component.
        y[idx] = X1; // Horizontal y-component.
        z[idx] = Y2; // Horizontal z-component.
    }
}

// Double precision.
__global__
void oskar_convert_apparent_ha_dec_to_enu_direction_cosines_cudak_d(int n,
        const double* ha, const double* dec, double cosLat, double sinLat,
        double* x, double* y, double* z)
{
    // Get the coordinate index that this thread is working on.
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Copy local equatorial coordinates from global memory.
    double sh, sd; // Source HA, Dec.
    if (idx < n)
    {
        sh = ha[idx];
        sd = dec[idx];
    }
    __syncthreads(); // Coalesce memory accesses.

    // Find direction cosines.
    double cosDec, sinDec, cosHA, sinHA, t, X1, Y2;
    sincos(sh, &sinHA, &cosHA);
    sincos(sd, &sinDec, &cosDec);
    t = cosDec * cosHA;
    X1 = cosLat * sinDec - sinLat * t;
    Y2 = sinLat * sinDec + cosLat * t;
    t = -cosDec * sinHA;

    // Copy direction cosines into global memory.
    __syncthreads(); // Coalesce memory accesses.
    if (idx < n)
    {
        x[idx] = t;  // Horizontal x-component.
        y[idx] = X1; // Horizontal y-component.
        z[idx] = Y2; // Horizontal z-component.
    }
}

#ifdef __cplusplus
}
#endif
