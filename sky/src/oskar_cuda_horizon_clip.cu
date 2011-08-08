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

int oskar_cuda_horizon_clip_f(const oskar_SkyModelGlobal_f* global, float lst,
        float lat, oskar_SkyModelLocal_f* local)
{
    // Extract pointers out of the structures.
    int n_in = global->num_sources;
    const float* in_I = global->I;
    const float* in_Q = global->Q;
    const float* in_U = global->U;
    const float* in_V = global->V;
    const float* in_ra = global->RA;
    const float* in_dec = global->Dec;
    int* n_out = &local->num_sources;
    float* out_I = local->I;
    float* out_Q = local->Q;
    float* out_U = local->U;
    float* out_V = local->V;
    float* out_ra = local->RA;
    float* out_dec = local->Dec;
    float* hor_l = local->hor_l;
    float* hor_m = local->hor_m;
    float* hor_n = local->hor_n;

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

int oskar_cuda_horizon_clip_d(const oskar_SkyModelGlobal_d* global, double lst,
        double lat, oskar_SkyModelLocal_d* local)
{
    // Extract pointers out of the structures.
    int n_in = global->num_sources;
    const double* in_I = global->I;
    const double* in_Q = global->Q;
    const double* in_U = global->U;
    const double* in_V = global->V;
    const double* in_ra = global->RA;
    const double* in_dec = global->Dec;
    int* n_out = &local->num_sources;
    double* out_I = local->I;
    double* out_Q = local->Q;
    double* out_U = local->U;
    double* out_V = local->V;
    double* out_ra = local->RA;
    double* out_dec = local->Dec;
    double* hor_l = local->hor_l;
    double* hor_m = local->hor_m;
    double* hor_n = local->hor_n;

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
