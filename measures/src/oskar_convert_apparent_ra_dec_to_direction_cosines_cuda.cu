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


#include "oskar_convert_apparent_ra_dec_to_direction_cosines_cuda.h"

/*#include "oskar_compute_tangent_plane_direction_z_cuda.h"
#include "oskar_convert_lon_lat_to_tangent_plane_direction_cuda.h"*/

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.
__global__
void oskar_convert_apparent_ra_dec_to_direction_cosines_cudak_f(const int n,
        const float* ra, const float* dec, const float ra0,
        const float cosDec0, const float sinDec0, float* x, float* y, float* z)
{
    // Get the position ID that this thread is working on.
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Copy the input data from global memory.
    float cosLat, sinLat, sinLon, cosLon, relLon, pLat;
    if (i < n)
    {
        relLon = ra[i];
        pLat = dec[i];
    }

    // Convert from spherical to tangent-plane.
    relLon -= ra0;
    sincosf(relLon, &sinLon, &cosLon);
    sincosf(pLat, &sinLat, &cosLat);
    float l = cosLat * sinLon;
    float m = cosDec0 * sinLat;
    m -= sinDec0 * cosLat * cosLon;

    // Output data.
    if (i < n)
    {
        x[i] = l;
        y[i] = m;
        float a = 1.0f - l*l - m*m;
        // FIXME this is z-1
        if (a < 0.0f)
            z[i] = -1.0f;
        else
            z[i] = sqrtf(a) - 1.0f;
    }
}

// Double precision.
__global__
void oskar_convert_apparent_ra_dec_to_direction_cosines_cudak_d(const int n,
        const double* ra, const double* dec, const double ra0,
        const double cosDec0, const double sinDec0, double* x, double* y,
        double* z)
{
    // Get the position ID that this thread is working on.
    const int i = blockDim.x * blockIdx.x + threadIdx.x;

    // Copy the input data from global memory.
    double cosLat, sinLat, sinLon, cosLon, relLon, pLat;
    if (i < n)
    {
        relLon = ra[i];
        pLat = dec[i];
    }

    // Convert from spherical to tangent-plane.
    relLon -= ra0;
    sincos(relLon, &sinLon, &cosLon);
    sincos(pLat, &sinLat, &cosLat);
    double l = cosLat * sinLon;
    double m = cosDec0 * sinLat;
    m -= sinDec0 * cosLat * cosLon;

    // Output data.
    if (i < n)
    {
        x[i] = l;
        y[i] = m;
        double a = 1.0 - l*l - m*m;
        // FIXME this is z-1
        if (a < 0.0)
            z[i] = -1.0;
        else
            z[i] = sqrt(a) - 1.0;
    }
}

/* Single precision. */
void oskar_convert_apparent_ra_dec_to_direction_cosines_cuda_f(
        int num_points, const float* ra, const float* dec, float ra0,
        float dec0, float* x, float* y, float* z)
{
    float cosDec0, sinDec0;
    int num_blocks, num_threads = 256;

    /* Compute x,y,z-direction-cosines of RA, Dec relative to reference point. */
    num_blocks = (num_points + num_threads - 1) / num_threads;
    cosDec0 = cosf(dec0);
    sinDec0 = sinf(dec0);

    oskar_convert_apparent_ra_dec_to_direction_cosines_cudak_f
        OSKAR_CUDAK_CONF(num_blocks, num_threads)
        (num_points, ra, dec, ra0, cosDec0, sinDec0, x, y, z);

    /*oskar_convert_lon_lat_to_tangent_plane_direction_cudak_f
            OSKAR_CUDAK_CONF(num_blocks, num_threads) (
            num_points, ra, dec, ra0, cosDec0, sinDec0, x, y);*/

    /* Compute z-direction-cosines of points from x and y. */
    /*oskar_compute_tangent_plane_direction_z_cudak_f
        OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, x, y, z);*/
}

/* Double precision. */
void oskar_convert_apparent_ra_dec_to_direction_cosines_cuda_d(
        int num_points, const double* ra, const double* dec, double ra0,
        double dec0, double* x, double* y, double* z)
{
    double cosDec0, sinDec0;
    int num_blocks, num_threads = 256;

    /* Compute x,y,z-direction-cosines of RA, Dec relative to reference point. */
    num_blocks = (num_points + num_threads - 1) / num_threads;
    cosDec0 = cos(dec0);
    sinDec0 = sin(dec0);

    oskar_convert_apparent_ra_dec_to_direction_cosines_cudak_d
        OSKAR_CUDAK_CONF(num_blocks, num_threads)
        (num_points, ra, dec, ra0, cosDec0, sinDec0, x, y, z);

    /*oskar_convert_lon_lat_to_tangent_plane_direction_cudak_d
            OSKAR_CUDAK_CONF(num_blocks, num_threads) (
            num_points, ra, dec, ra0, cosDec0, sinDec0, x, y);*/

    /* Compute z-direction-cosines of points from x and y. */
    /*oskar_compute_tangent_plane_direction_z_cudak_d
        OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, x, y, z);*/
}



#ifdef __cplusplus
}
#endif
