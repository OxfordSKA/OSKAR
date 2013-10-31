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

#include <oskar_convert_apparent_ra_dec_to_enu_direction_cosines_cuda.h>
#include <oskar_convert_apparent_ha_dec_to_enu_direction_cosines_cuda.h>
#include <math/cudak/oskar_cudak_vec_sub_sr.h>

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.
void oskar_convert_apparent_ra_dec_to_enu_direction_cosines_cuda_f(int n,
        const float* ra, const float* dec, float lst, float lat, float* x,
        float* y, float* z)
{
    // Determine Hour Angles(HA = LST - RA).
    float* ha = z; // Temporary.
    const int n_thd = 256;
    const int n_blk_in = (n + n_thd - 1) / n_thd;
    oskar_cudak_vec_sub_sr_f OSKAR_CUDAK_CONF(n_blk_in, n_thd)(n, lst, ra, ha);

    // Determine horizontal x,y,z positions (destroys contents of ha).
    float cosLat = cosf(lat);
    float sinLat = sinf(lat);
    oskar_convert_apparent_ha_dec_to_enu_direction_cosines_cudak_f
    OSKAR_CUDAK_CONF(n_blk_in, n_thd) (n, ha, dec, cosLat, sinLat, x, y, z);
}

// Double precision.
void oskar_convert_apparent_ra_dec_to_enu_direction_cosines_cuda_d(int n,
        const double* ra, const double* dec, double lst, double lat, double* x,
        double* y, double* z)
{
    // Determine source Hour Angles (HA = LST - RA).
    double* ha = z; // Temporary.
    const int n_thd = 256;
    const int n_blk_in = (n + n_thd - 1) / n_thd;
    oskar_cudak_vec_sub_sr_d OSKAR_CUDAK_CONF(n_blk_in, n_thd)(n, lst, ra, ha);

    // Determine horizontal l,m,n positions (destroys contents of ha).
    double cosLat = cos(lat);
    double sinLat = sin(lat);
    oskar_convert_apparent_ha_dec_to_enu_direction_cosines_cudak_d
    OSKAR_CUDAK_CONF(n_blk_in, n_thd) (n, ha, dec, cosLat, sinLat, x, y, z);
}


// Single precision.
__global__
void oskar_convert_apparent_ra_dec_to_enu_direction_cosines_cudak_f(int n,
        const float* ra, const float* dec, float cosLat, float sinLat,
        float lst, float* x, float* y, float* z)
{
    // Get the coordinate index that this thread is working on.
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Copy local equatorial coordinates from global memory.
    float sh, sd; // HA, Dec.
    if (idx < n)
    {
        sh = ra[idx];
        sd = dec[idx];
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
void oskar_convert_apparent_ra_dec_to_enu_direction_cosines_cudak_d(int n,
        const double* ra, const double* dec, double cosLat, double sinLat,
        double lst, double* x, double* y, double* z)
{
    // Get the coordinate index that this thread is working on.
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    // Copy local equatorial coordinates from global memory.
    double sh, sd; // Source HA, Dec.
    if (idx < n)
    {
        sh = ra[idx];
        sd = dec[idx];
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
