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

#include "sky/cudak/oskar_cudak_ra_dec_to_hor_lmn.h"

// Single precision.

__global__
void oskar_cudak_ra_dec_to_hor_lmn_f(int ns, const float* ra,
        const float* dec, float cosLat, float sinLat, float lst,
        float* l, float* m, float* n)
{
    // Get the source ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    // Copy source local equatorial coordinates from global memory.
    float sh, sd; // Source HA, Dec.
    if (s < ns)
    {
        sh = ra[s];
        sd = dec[s];
    }
    __syncthreads(); // Coalesce memory accesses.
    sh = lst - sh; // HA = LST - RA.

    // Find direction cosines.
    float cosDec, sinDec, cosHA, sinHA, t, X1, Y2;
    sincosf(sh, &sinHA, &cosHA);
    sincosf(sd, &sinDec, &cosDec);
    t = cosDec * cosHA;
    X1 = cosLat * sinDec - sinLat * t;
    Y2 = sinLat * sinDec + cosLat * t;
    t = -cosDec * sinHA;

    // Copy source direction cosines into global memory.
    __syncthreads(); // Coalesce memory accesses.
    if (s < ns)
    {
        l[s] = t;  // Horizontal x-component.
        m[s] = X1; // Horizontal y-component.
        n[s] = Y2; // Horizontal z-component.
    }
}

// Double precision.

__global__
void oskar_cudak_ra_dec_to_hor_lmn_d(int ns, const double* ra,
        const double* dec, double cosLat, double sinLat, double lst,
        double* l, double* m, double* n)
{
    // Get the source ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    // Copy source local equatorial coordinates from global memory.
    double sh, sd; // Source HA, Dec.
    if (s < ns)
    {
        sh = ra[s];
        sd = dec[s];
    }
    __syncthreads(); // Coalesce memory accesses.
    sh = lst - sh; // HA = LST - RA.

    // Find direction cosines.
    double cosDec, sinDec, cosHA, sinHA, t, X1, Y2;
    sincos(sh, &sinHA, &cosHA);
    sincos(sd, &sinDec, &cosDec);
    t = cosDec * cosHA;
    X1 = cosLat * sinDec - sinLat * t;
    Y2 = sinLat * sinDec + cosLat * t;
    t = -cosDec * sinHA;

    // Copy source direction cosines into global memory.
    __syncthreads(); // Coalesce memory accesses.
    if (s < ns)
    {
        l[s] = t;  // Horizontal x-component.
        m[s] = X1; // Horizontal y-component.
        n[s] = Y2; // Horizontal z-component.
    }
}
