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
#include "sky/oskar_sky_cuda_ra_dec_to_hor_lmn.h"
#include "sky/cudak/oskar_sky_cudak_ha_dec_to_hor_lmn.h"
#include "math/cudak/oskar_math_cudak_sph_to_lm.h"
#include "math/cudak/oskar_math_cudak_vec_sub_sr.h"

#include <thrust/device_vector.h> // Must be included before thrust/copy.h
#include <thrust/copy.h>

// Single precision.

struct is_positive_f {
    __host__ __device__
    bool operator()(const float x) {return x > 0.0f;}
};

int oskar_sky_cudaf_horizon_clip(int n_in, const float* in_I,
        const float* in_Q, const float* in_U, const float* in_V,
        const float* ra, const float* dec, float ra0, float dec0,
        float lst, float lat, int* n_out, float* out_I, float* out_Q,
        float* out_U, float* out_V, float* eq_l, float* eq_m,
        float* hor_l, float* hor_m, float* work)
{
    // Determine horizontal l,m,n positions (temporaries in eq_l, eq_m).
    float* hor_n = work;
    int rv = oskar_sky_cudaf_ra_dec_to_hor_lmn
            (n_in, ra, dec, lst, lat, eq_l, eq_m, hor_n);
    if (rv) return rv;

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
            thrust::device_pointer_cast(in_I),         // Input start.
            thrust::device_pointer_cast(in_I + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(out_I), is_positive_f());
    thrust::copy_if(
            thrust::device_pointer_cast(in_Q),         // Input start.
            thrust::device_pointer_cast(in_Q + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(out_Q), is_positive_f());
    thrust::copy_if(
            thrust::device_pointer_cast(in_U),         // Input start.
            thrust::device_pointer_cast(in_U + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(out_U), is_positive_f());
    thrust::copy_if(
            thrust::device_pointer_cast(in_V),         // Input start.
            thrust::device_pointer_cast(in_V + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(out_V), is_positive_f());
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
    const int n_thd = 256;
    const int n_blk = (*n_out + n_thd - 1) / n_thd;
    const float cosDec0 = cosf(dec0);
    const float sinDec0 = sinf(dec0);
    oskar_math_cudakf_sph_to_lm <<< n_blk, n_thd >>>
            (*n_out, eq_l, eq_m, ra0, cosDec0, sinDec0, eq_l, eq_m);
    cudaDeviceSynchronize();
    cudaError_t errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) return errCuda;

    return 0;
}

// Double precision.

struct is_positive_d {
    __host__ __device__
    bool operator()(const double x) {return x > 0.0;}
};

int oskar_sky_cudad_horizon_clip(int n_in, const double* in_I,
        const double* in_Q, const double* in_U, const double* in_V,
        const double* ra, const double* dec, double ra0, double dec0,
        double lst, double lat, int* n_out, double* out_I, double* out_Q,
        double* out_U, double* out_V, double* eq_l, double* eq_m,
        double* hor_l, double* hor_m, double* work)
{
    // Determine horizontal l,m,n positions (temporaries in eq_l, eq_m).
    double* hor_n = work;
    int rv = oskar_sky_cudad_ra_dec_to_hor_lmn
            (n_in, ra, dec, lst, lat, eq_l, eq_m, hor_n);
    if (rv) return rv;

    // Determine which sources are above the horizon, and copy them out.
    thrust::device_ptr<double> out = thrust::copy_if(
            thrust::device_pointer_cast(eq_l),         // Input start.
            thrust::device_pointer_cast(eq_l + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(hor_l), is_positive_d());
    thrust::copy_if(
            thrust::device_pointer_cast(eq_m),         // Input start.
            thrust::device_pointer_cast(eq_m + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(hor_m), is_positive_d());
    thrust::copy_if(
            thrust::device_pointer_cast(in_I),         // Input start.
            thrust::device_pointer_cast(in_I + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(out_I), is_positive_d());
    thrust::copy_if(
            thrust::device_pointer_cast(in_Q),         // Input start.
            thrust::device_pointer_cast(in_Q + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(out_Q), is_positive_d());
    thrust::copy_if(
            thrust::device_pointer_cast(in_U),         // Input start.
            thrust::device_pointer_cast(in_U + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(out_U), is_positive_d());
    thrust::copy_if(
            thrust::device_pointer_cast(in_V),         // Input start.
            thrust::device_pointer_cast(in_V + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(out_V), is_positive_d());
    thrust::copy_if(
            thrust::device_pointer_cast(ra),           // Input start.
            thrust::device_pointer_cast(ra + n_in),    // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(eq_l), is_positive_d());
    thrust::copy_if(
            thrust::device_pointer_cast(dec),          // Input start.
            thrust::device_pointer_cast(dec + n_in),   // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil
            thrust::device_pointer_cast(eq_m), is_positive_d());

    // Compute l,m direction cosines of visible source RA, Dec
    // relative to phase centre.
    *n_out = out - thrust::device_pointer_cast(hor_l);
    const int n_thd = 256;
    const int n_blk = (*n_out + n_thd - 1) / n_thd;
    const double cosDec0 = cos(dec0);
    const double sinDec0 = sin(dec0);
    oskar_math_cudakd_sph_to_lm <<< n_blk, n_thd >>>
            (*n_out, eq_l, eq_m, ra0, cosDec0, sinDec0, eq_l, eq_m);
    cudaDeviceSynchronize();
    cudaError_t errCuda = cudaPeekAtLastError();
    if (errCuda != cudaSuccess) return errCuda;

    return 0;
}
