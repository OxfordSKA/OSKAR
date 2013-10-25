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

#include "oskar_convert_lon_lat_to_tangent_plane_direction_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

// Single precision.
__global__
void oskar_convert_lon_lat_to_tangent_plane_direction_cudak_f(const int np,
        const float* lon, const float* lat, const float lon0,
        const float cosLat0, const float sinLat0, float* x, float* y)
{
    // Get the position ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    // Copy the input data from global memory.
    float cosPhi, sinPhi, sinLambda, cosLambda, relLambda, pphi;
    if (s < np)
    {
        relLambda = lon[s];
        pphi = lat[s];
    }

    // Convert from spherical to tangent-plane.
    relLambda -= lon0;
    sincosf(relLambda, &sinLambda, &cosLambda);
    sincosf(pphi, &sinPhi, &cosPhi);
    float ll = cosPhi * sinLambda;
    float mm = cosLat0 * sinPhi;
    mm -= sinLat0 * cosPhi * cosLambda;

    // Output data.
    if (s < np)
    {
        x[s] = ll;
        y[s] = mm;
    }
}

// Double precision.
__global__
void oskar_convert_lon_lat_to_tangent_plane_direction_cudak_d(const int np,
        const double* lon, const double* lat, const double lon0,
        const double cosLat0, const double sinLat0, double* x, double* y)
{
    // Get the position ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    // Copy the input data from global memory.
    double cosPhi, sinPhi, sinLambda, cosLambda, relLambda, pphi;
    if (s < np)
    {
        relLambda = lon[s];
        pphi = lat[s];
    }

    // Convert from spherical to tangent-plane.
    relLambda -= lon0;
    sincos(relLambda, &sinLambda, &cosLambda);
    sincos(pphi, &sinPhi, &cosPhi);
    double ll = cosPhi * sinLambda;
    double mm = cosLat0 * sinPhi;
    mm -= sinLat0 * cosPhi * cosLambda;

    // Output data.
    if (s < np)
    {
        x[s] = ll;
        y[s] = mm;
    }
}


#ifdef __cplusplus
}
#endif
