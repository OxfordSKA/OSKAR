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

#include "sky/oskar_sky_cuda_horizon_clip.h"
#include "sky/cudak/oskar_sky_cudak_ha_dec_to_hor_lmn.h"
#include "math/cudak/oskar_math_cudak_sph_to_lm.h"
#include "math/cudak/oskar_math_cudak_vec_sub_sr.h"

#include <thrust/copy.h>

// Single precision.

static struct is_positive_f {
    __host__ __device__
    bool operator()(const float x) {return x > 0.0f;}
};

int oskar_sky_cudaf_horizon_clip(int n_in, const float* in_b,
        const float* ra, const float* dec, float ra0, float dec0,
        float lst, float lat, int* n_out, float* out_b, float* eq_l,
        float* eq_m, float* hor_l, float* hor_m, float* work)
{
    // Determine source Hour Angles (HA = LST - RA).
    float* ha = work;
    const int n_thd = 256;
    const int n_blk_in = (n_in + n_thd - 1) / n_thd;
    oskar_math_cudakf_vec_sub_sr <<< n_blk_in, n_thd >>> (n_in, lst, ra, ha);

    // Determine horizontal l,m,n positions (destroys contents of ha).
    float cosLat = cosf(lat);
    float sinLat = sinf(lat);
    // Temporarily use work for hor_n and eq_l, eq_m for hor_l, hor_m.
    float* hor_n = work;
    oskar_sky_cudakf_ha_dec_to_hor_lmn <<< n_blk_in, n_thd >>> (n_in, ha, dec,
            cosLat, sinLat, eq_l, eq_m, hor_n);
    cudaDeviceSynchronize();
    cudaError_t errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) return errCuda;

    // Determine which sources are above the horizon, and copy them out.
    thrust::device_ptr<float> out = thrust::copy_if(
            thrust::device_pointer_cast(eq_l),         // Input start.
            thrust::device_pointer_cast(eq_l + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(hor_l), is_positive_f());
    thrust::copy_if(
            thrust::device_pointer_cast(eq_m),         // Input start.
            thrust::device_pointer_cast(eq_m + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(hor_m), is_positive_f());
    thrust::copy_if(
            thrust::device_pointer_cast(in_b),         // Input start.
            thrust::device_pointer_cast(in_b + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(out_b), is_positive_f());
    thrust::copy_if(
            thrust::device_pointer_cast(ra),           // Input start.
            thrust::device_pointer_cast(ra + n_in),    // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(eq_l), is_positive_f());
    thrust::copy_if(
            thrust::device_pointer_cast(dec),          // Input start.
            thrust::device_pointer_cast(dec + n_in),   // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(eq_m), is_positive_f());

    // Compute l,m direction cosines of visible source RA, Dec
    // relative to phase centre.
    *n_out = out - thrust::device_pointer_cast(hor_l);
    const int n_blk_out = (*n_out + n_thd - 1) / n_thd;
    const float cosDec0 = cosf(dec0);
    const float sinDec0 = sinf(dec0);
    oskar_math_cudakf_sph_to_lm <<< n_blk_out, n_thd >>>
            (*n_out, eq_l, eq_m, ra0, cosDec0, sinDec0, eq_l, eq_m);
    cudaDeviceSynchronize();
    errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) return errCuda;

    return 0;
}

// Double precision.

static struct is_positive_d {
    __host__ __device__
    bool operator()(const double x) {return x > 0.0;}
};

int oskar_sky_cudad_horizon_clip(int n_in, const double* in_b,
        const double* ra, const double* dec, double ra0, double dec0,
        double lst, double lat, int* n_out, double* out_b, double* eq_l,
        double* eq_m, double* hor_l, double* hor_m, double* work)
{
    // Determine source Hour Angles (HA = LST - RA).
    double* ha = work;
    const int n_thd = 256;
    const int n_blk_in = (n_in + n_thd - 1) / n_thd;
    oskar_math_cudakf_vec_sub_sr <<< n_blk_in, n_thd >>> (n_in, lst, ra, ha);

    // Determine horizontal l,m,n positions (destroys contents of ha).
    double cosLat = cos(lat);
    double sinLat = sin(lat);
    // Temporarily use work for hor_n and eq_l, eq_m for hor_l, hor_m.
    double* hor_n = work;
    oskar_sky_cudakf_ha_dec_to_hor_lmn <<< n_blk_in, n_thd >>> (n_in, ha, dec,
            cosLat, sinLat, eq_l, eq_m, hor_n);
    cudaDeviceSynchronize();
    cudaError_t errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) return errCuda;

    // Determine which sources are above the horizon, and copy them out.
    thrust::device_ptr<double> out = thrust::copy_if(
            thrust::device_pointer_cast(eq_l),         // Input start.
            thrust::device_pointer_cast(eq_l + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(hor_l), is_positive_f());
    thrust::copy_if(
            thrust::device_pointer_cast(eq_m),         // Input start.
            thrust::device_pointer_cast(eq_m + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(hor_m), is_positive_f());
    thrust::copy_if(
            thrust::device_pointer_cast(in_b),         // Input start.
            thrust::device_pointer_cast(in_b + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(out_b), is_positive_f());
    thrust::copy_if(
            thrust::device_pointer_cast(ra),           // Input start.
            thrust::device_pointer_cast(ra + n_in),    // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(eq_l), is_positive_f());
    thrust::copy_if(
            thrust::device_pointer_cast(dec),          // Input start.
            thrust::device_pointer_cast(dec + n_in),   // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(eq_m), is_positive_f());

    // Compute l,m direction cosines of visible source RA, Dec
    // relative to phase centre.
    *n_out = out - thrust::device_pointer_cast(hor_l);
    const int n_blk_out = (*n_out + n_thd - 1) / n_thd;
    const double cosDec0 = cos(dec0);
    const double sinDec0 = sin(dec0);
    oskar_math_cudakf_sph_to_lm <<< n_blk_out, n_thd >>>
            (*n_out, eq_l, eq_m, ra0, cosDec0, sinDec0, eq_l, eq_m);
    cudaDeviceSynchronize();
    errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) return errCuda;

    return 0;
}
