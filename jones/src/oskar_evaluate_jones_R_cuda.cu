/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include <oskar_evaluate_jones_R_cuda.h>
#include <oskar_parallactic_angle.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_evaluate_jones_R_cuda_f(float4c* d_jones, int num_sources,
        const float* d_ra_rad, const float* d_dec_rad, float latitude_rad,
        float lst_rad)
{
    float cos_lat, sin_lat;
    int num_blocks, num_threads = 256;

    /* Set up thread blocks. */
    num_blocks = (num_sources + num_threads - 1) / num_threads;

    /* Evaluate parallactic angle rotation for all sources. */
    cos_lat = cos(latitude_rad);
    sin_lat = sin(latitude_rad);
    oskar_evaluate_jones_R_cudak_f OSKAR_CUDAK_CONF(num_blocks, num_threads)
            (d_jones, num_sources, d_ra_rad, d_dec_rad, cos_lat, sin_lat,
                    lst_rad);
}

/* Double precision. */
void oskar_evaluate_jones_R_cuda_d(double4c* d_jones, int num_sources,
        const double* d_ra_rad, const double* d_dec_rad, double latitude_rad,
        double lst_rad)
{
    double cos_lat, sin_lat;
    int num_blocks, num_threads = 256;

    /* Set up thread blocks. */
    num_blocks = (num_sources + num_threads - 1) / num_threads;

    /* Evaluate parallactic angle rotation for all sources. */
    cos_lat = cos(latitude_rad);
    sin_lat = sin(latitude_rad);
    oskar_evaluate_jones_R_cudak_d OSKAR_CUDAK_CONF(num_blocks, num_threads)
            (d_jones, num_sources, d_ra_rad, d_dec_rad, cos_lat, sin_lat,
                    lst_rad);
}

#ifdef __cplusplus
}
#endif


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_evaluate_jones_R_cudak_f(float4c* jones, const int num_sources,
        const float* ra_rad, const float* dec_rad, const float cos_lat,
        const float sin_lat, const float lst_rad)
{
    float c_ha, c_dec, q, sin_q, cos_q;
    float4c J;

    /* Get the source ID that this thread is working on. */
    const int s = blockDim.x * blockIdx.x + threadIdx.x;
    if (s >= num_sources) return;

    /* Copy the data from global memory. */
    c_ha = ra_rad[s]; /* Source RA, but will be source hour angle. */
    c_dec = dec_rad[s];

    /* Compute the source hour angle. */
    c_ha = lst_rad - c_ha; /* HA = LST - RA. */

    /* Compute the source parallactic angle. */
    q = oskar_parallactic_angle_f(c_ha, c_dec, cos_lat, sin_lat);
    sincosf(q, &sin_q, &cos_q);

    /* Compute the Jones matrix. */
    J.a = make_float2(cos_q, 0.0f);
    J.b = make_float2(-sin_q, 0.0f);
    J.c = make_float2(sin_q, 0.0f);
    J.d = make_float2(cos_q, 0.0f);

    /* Copy the Jones matrix to global memory. */
    jones[s] = J;
}

/* Double precision. */
__global__
void oskar_evaluate_jones_R_cudak_d(double4c* jones, int num_sources,
        const double* ra_rad, const double* dec_rad, const double cos_lat,
        const double sin_lat, const double lst_rad)
{
    double c_ha, c_dec, q, sin_q, cos_q;
    double4c J;

    /* Get the source ID that this thread is working on. */
    const int s = blockDim.x * blockIdx.x + threadIdx.x;
    if (s >= num_sources) return;

    /* Copy the data from global memory. */
    c_ha = ra_rad[s]; /* Source RA, but will be source hour angle. */
    c_dec = dec_rad[s];

    /* Compute the source hour angle. */
    c_ha = lst_rad - c_ha; /* HA = LST - RA. */

    /* Compute the source parallactic angle. */
    q = oskar_parallactic_angle_d(c_ha, c_dec, cos_lat, sin_lat);
    sincos(q, &sin_q, &cos_q);

    /* Compute the Jones matrix. */
    J.a = make_double2(cos_q, 0.0);
    J.b = make_double2(-sin_q, 0.0);
    J.c = make_double2(sin_q, 0.0);
    J.d = make_double2(cos_q, 0.0);

    /* Copy the Jones matrix to global memory. */
    jones[s] = J;
}
