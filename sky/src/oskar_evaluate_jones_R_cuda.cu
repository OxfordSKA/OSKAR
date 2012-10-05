/*
 * Copyright (c) 2012, The University of Oxford
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

#include "sky/oskar_evaluate_jones_R_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_evaluate_jones_R_cuda_f(float4c* d_jones, int num_sources,
        float* d_ra, float* d_dec, float latitude_rad, float lst_rad)
{
    float cos_lat, sin_lat;
    int num_blocks, num_threads = 256;

    /* Set up thread blocks. */
    num_blocks = (num_sources + num_threads - 1) / num_threads;

    /* Evaluate parallactic angle rotation for all sources. */
    cos_lat = cos(latitude_rad);
    sin_lat = sin(latitude_rad);
    oskar_evaluate_jones_R_cudak_f OSKAR_CUDAK_CONF(num_blocks, num_threads)
            (d_jones, num_sources, d_ra, d_dec, cos_lat, sin_lat, lst_rad);
}

/* Double precision. */
void oskar_evaluate_jones_R_cuda_d(double4c* d_jones, int num_sources,
        double* d_ra, double* d_dec, double latitude_rad, double lst_rad)
{
    double cos_lat, sin_lat;
    int num_blocks, num_threads = 256;

    /* Set up thread blocks. */
    num_blocks = (num_sources + num_threads - 1) / num_threads;

    /* Evaluate parallactic angle rotation for all sources. */
    cos_lat = cos(latitude_rad);
    sin_lat = sin(latitude_rad);
    oskar_evaluate_jones_R_cudak_d OSKAR_CUDAK_CONF(num_blocks, num_threads)
            (d_jones, num_sources, d_ra, d_dec, cos_lat, sin_lat, lst_rad);
}

#ifdef __cplusplus
}
#endif


/* Kernels. ================================================================ */

/**
 * @brief
 * CUDA device function to compute the parallactic angle at a position
 * (single precision)
 *
 * @details
 * This inline device function computes the parallactic angle at a position
 * on the sky, at a given latitude.
 *
 * @param[in] ha       The hour angle.
 * @param[in] dec      The declination.
 * @param[in] cos_lat  Cosine of the observer's latitude.
 * @param[in] sin_lat  Sine of the observer's latitude.
 */
__device__ __forceinline__
static float oskar_parallactic_angle_f(const float& ha,
        const float& dec, const float& cos_lat, const float& sin_lat)
{
    float sin_dec, cos_dec, sin_a, cos_a;
    sincosf(ha, &sin_a, &cos_a);
    sincosf(dec, &sin_dec, &cos_dec);
    float y = cos_lat * sin_a;
    float x = sin_lat * cos_dec - cos_lat * sin_dec * cos_a;
    return atan2f(y, x);
}

/**
 * @brief
 * CUDA device function to compute the parallactic angle at a position
 * (double precision)
 *
 * @details
 * This inline device function computes the parallactic angle at a position
 * on the sky, at a given latitude.
 *
 * @param[in] ha       The hour angle.
 * @param[in] dec      The declination.
 * @param[in] cos_lat  Cosine of the observer's latitude.
 * @param[in] sin_lat  Sine of the observer's latitude.
 */
__device__ __forceinline__
static double oskar_parallactic_angle_d(const double& ha,
        const double& dec, const double& cos_lat, const double& sin_lat)
{
    double sin_dec, cos_dec, sin_a, cos_a;
    sincos(ha, &sin_a, &cos_a);
    sincos(dec, &sin_dec, &cos_dec);
    double y = cos_lat * sin_a;
    double x = sin_lat * cos_dec - cos_lat * sin_dec * cos_a;
    return atan2(y, x);
}


/* Single precision. */
__global__
void oskar_evaluate_jones_R_cudak_f(float4c* jones, const int num_sources,
        const float* ra, const float* dec, const float cos_lat,
        const float sin_lat, const float lst_rad)
{
    /* Get the source ID that this thread is working on. */
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    /* Copy the data from global memory. */
    float c_ha, c_dec;
    if (s < num_sources)
    {
        c_ha = ra[s]; /* Source RA, but will be source hour angle. */
        c_dec = dec[s];
    }

    /* Compute the source hour angle. */
    c_ha = lst_rad - c_ha; /* HA = LST - RA. */

    /* Compute the source parallactic angle. */
    float sin_a, cos_a;
    {
        float q = oskar_parallactic_angle_f(c_ha, c_dec, cos_lat, sin_lat);
        sincosf(q, &sin_a, &cos_a);
    }

    /* Compute the Jones matrix. */
    float4c J;
    J.a = make_float2(cos_a, 0.0f);
    J.b = make_float2(-sin_a, 0.0f);
    J.c = make_float2(sin_a, 0.0f);
    J.d = make_float2(cos_a, 0.0f);

    /* Copy the Jones matrix to global memory. */
    if (s < num_sources)
        jones[s] = J;
}

/* Double precision. */
__global__
void oskar_evaluate_jones_R_cudak_d(double4c* jones, int num_sources,
        const double* ra, const double* dec, const double cos_lat,
        const double sin_lat, const double lst_rad)
{
    /* Get the source ID that this thread is working on. */
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    /* Copy the data from global memory. */
    double c_ha, c_dec;
    if (s < num_sources)
    {
        c_ha = ra[s]; /* Source RA, but will be source hour angle. */
        c_dec = dec[s];
    }

    /* Compute the source hour angle. */
    c_ha = lst_rad - c_ha; /* HA = LST - RA. */

    /* Compute the source parallactic angle. */
    double sin_a, cos_a;
    {
        double q = oskar_parallactic_angle_d(c_ha, c_dec, cos_lat, sin_lat);
        sincos(q, &sin_a, &cos_a);
    }

    /* Compute the Jones matrix. */
    double4c J;
    J.a = make_double2(cos_a, 0.0);
    J.b = make_double2(-sin_a, 0.0);
    J.c = make_double2(sin_a, 0.0);
    J.d = make_double2(cos_a, 0.0);

    /* Copy the Jones matrix to global memory. */
    if (s < num_sources)
        jones[s] = J;
}
