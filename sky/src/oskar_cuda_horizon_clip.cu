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

#include "sky/oskar_cuda_horizon_clip.h"
#include "sky/oskar_cuda_ra_dec_to_hor_lmn.h"
#include "sky/oskar_cuda_ra_dec_to_relative_lmn.h"

#include <thrust/device_vector.h> // Must be included before thrust/copy.h
#include <thrust/copy.h>
#include <thrust/remove.h>

// Single precision.

struct is_positive_f {
    __host__ __device__
    bool operator()(const float x) {return x > 0.0f;}
};

struct is_negative_f {
    __host__ __device__
    bool operator()(const float x) {return x <= 0.0f;}
};

int oskar_cuda_horizon_clip_f(int n_in, const float* in_I,
        const float* in_Q, const float* in_U, const float* in_V,
        const float* in_ra, const float* in_dec, float lst, float lat,
        int* n_out, float* out_I, float* out_Q, float* out_U, float* out_V,
        float* out_ra, float* out_dec, float* hor_l, float* hor_m,
        float* hor_n)
{
    // Determine horizontal l,m,n positions.
    int rv = oskar_cuda_ra_dec_to_hor_lmn_f
            (n_in, in_ra, in_dec, lst, lat, hor_l, hor_m, hor_n);
    if (rv) return rv;

    // Determine which sources are above the horizon, and copy them out.
    thrust::device_ptr<float> out = thrust::copy_if(
            thrust::device_pointer_cast(in_I),         // Input start.
            thrust::device_pointer_cast(in_I + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil.
            thrust::device_pointer_cast(out_I), is_positive_f());
    thrust::copy_if(
            thrust::device_pointer_cast(in_Q),         // Input start.
            thrust::device_pointer_cast(in_Q + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil.
            thrust::device_pointer_cast(out_Q), is_positive_f());
    thrust::copy_if(
            thrust::device_pointer_cast(in_U),         // Input start.
            thrust::device_pointer_cast(in_U + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil.
            thrust::device_pointer_cast(out_U), is_positive_f());
    thrust::copy_if(
            thrust::device_pointer_cast(in_V),         // Input start.
            thrust::device_pointer_cast(in_V + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil.
            thrust::device_pointer_cast(out_V), is_positive_f());
    thrust::copy_if(
            thrust::device_pointer_cast(in_ra),        // Input start.
            thrust::device_pointer_cast(in_ra + n_in), // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil.
            thrust::device_pointer_cast(out_ra), is_positive_f());
    thrust::copy_if(
            thrust::device_pointer_cast(in_dec),       // Input start.
            thrust::device_pointer_cast(in_dec + n_in),// Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil.
            thrust::device_pointer_cast(out_dec), is_positive_f());
    thrust::remove_if(
            thrust::device_pointer_cast(hor_l),        // Input start.
            thrust::device_pointer_cast(hor_l + n_in), // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil.
            is_negative_f());
    thrust::remove_if(
            thrust::device_pointer_cast(hor_m),        // Input start.
            thrust::device_pointer_cast(hor_m + n_in), // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil.
            is_negative_f());

    // Compact the stencil last.
    thrust::remove_if(
            thrust::device_pointer_cast(hor_n),        // Input start.
            thrust::device_pointer_cast(hor_n + n_in), // Input end.
            is_negative_f());

    // Get the number of sources above the horizon.
    *n_out = out - thrust::device_pointer_cast(out_I);

    return 0;
}

// Double precision.

struct is_positive_d {
    __host__ __device__
    bool operator()(const double x) {return x > 0.0;}
};

struct is_negative_d {
    __host__ __device__
    bool operator()(const double x) {return x <= 0.0;}
};

int oskar_cuda_horizon_clip_d(int n_in, const double* in_I,
        const double* in_Q, const double* in_U, const double* in_V,
        const double* in_ra, const double* in_dec, double lst, double lat,
        int* n_out, double* out_I, double* out_Q, double* out_U, double* out_V,
        double* out_ra, double* out_dec, double* hor_l, double* hor_m,
        double* hor_n)
{
    // Determine horizontal l,m,n positions.
    int rv = oskar_cuda_ra_dec_to_hor_lmn_d
            (n_in, in_ra, in_dec, lst, lat, hor_l, hor_m, hor_n);
    if (rv) return rv;

    // Determine which sources are above the horizon, and copy them out.
    thrust::device_ptr<double> out = thrust::copy_if(
            thrust::device_pointer_cast(in_I),         // Input start.
            thrust::device_pointer_cast(in_I + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil.
            thrust::device_pointer_cast(out_I), is_positive_d());
    thrust::copy_if(
            thrust::device_pointer_cast(in_Q),         // Input start.
            thrust::device_pointer_cast(in_Q + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil.
            thrust::device_pointer_cast(out_Q), is_positive_d());
    thrust::copy_if(
            thrust::device_pointer_cast(in_U),         // Input start.
            thrust::device_pointer_cast(in_U + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil.
            thrust::device_pointer_cast(out_U), is_positive_d());
    thrust::copy_if(
            thrust::device_pointer_cast(in_V),         // Input start.
            thrust::device_pointer_cast(in_V + n_in),  // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil.
            thrust::device_pointer_cast(out_V), is_positive_d());
    thrust::copy_if(
            thrust::device_pointer_cast(in_ra),        // Input start.
            thrust::device_pointer_cast(in_ra + n_in), // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil.
            thrust::device_pointer_cast(out_ra), is_positive_d());
    thrust::copy_if(
            thrust::device_pointer_cast(in_dec),       // Input start.
            thrust::device_pointer_cast(in_dec + n_in),// Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil.
            thrust::device_pointer_cast(out_dec), is_positive_d());
    thrust::remove_if(
            thrust::device_pointer_cast(hor_l),        // Input start.
            thrust::device_pointer_cast(hor_l + n_in), // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil.
            is_negative_d());
    thrust::remove_if(
            thrust::device_pointer_cast(hor_m),        // Input start.
            thrust::device_pointer_cast(hor_m + n_in), // Input end.
            thrust::device_pointer_cast(hor_n),        // Stencil.
            is_negative_d());

    // Compact the stencil last.
    thrust::remove_if(
            thrust::device_pointer_cast(hor_n),        // Input start.
            thrust::device_pointer_cast(hor_n + n_in), // Input end.
            is_negative_d());

    // Get the number of sources above the horizon.
    *n_out = out - thrust::device_pointer_cast(out_I);

    return 0;
}
