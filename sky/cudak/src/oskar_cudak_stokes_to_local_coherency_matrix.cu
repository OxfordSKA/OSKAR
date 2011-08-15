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

#include "sky/cudak/oskar_cudak_stokes_to_local_coherency_matrix.h"
#include "utility/oskar_vector_types.h"

// Single precision.

__global__
void oskar_cudak_stokes_to_local_coherency_matrix_f(int ns, const float* ra,
        const float* dec, const float* stokes_I, const float* stokes_Q,
        const float* stokes_U, const float* stokes_V, float cos_lat,
        float sin_lat, float lst, float4c* coherency_matrix)
{
    // Get the source ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    // Copy the data from global memory.
    float c_ha, c_dec, c_I, c_Q, c_U, c_V;
    if (s < ns)
    {
        c_ha = ra[s]; // Source RA, but will be source hour angle.
        c_dec = dec[s];
        c_I = stokes_I[s];
        c_Q = stokes_Q[s];
        c_U = stokes_U[s];
        c_V = stokes_V[s];
    }
    __syncthreads();

    // Compute the source hour angle.
    c_ha = lst - c_ha; // HA = LST - RA.

    // Compute the source parallactic angle.
    float sin_dec, cos_dec, sin_a, cos_a;
    sincosf(c_ha, &sin_a, &cos_a);
    sincosf(c_dec, &sin_dec, &cos_dec);
    float y = cos_lat * sin_a;
    float x = sin_lat * cos_dec - cos_lat * sin_dec * cos_a;
    float q = 2.0f * atan2f(y, x); // 2.0 * (parallactic angle)
    sincosf(q, &sin_a, &cos_a);

    // Compute the source coherency matrix.
    x = -c_Q * cos_a - c_U * sin_a; // Q = -Q' cos(2q) - U' sin(2q)
    y = -c_U * cos_a + c_Q * sin_a; // U = -U' cos(2q) + Q' sin(2q)
    float4c b;
    b.a.x = c_I + x; // I + Q
    b.a.y = 0.0f;
    b.b.x = y; // U + iV
    b.b.y = c_V;
    b.c.x = y; // U - iV
    b.c.y = -c_V;
    b.d.x = c_I - x; // I - Q
    b.d.y = 0.0f;

    // Copy the source coherency matrix to global memory.
    __syncthreads();
    if (s < ns)
        coherency_matrix[s] = b;
}

// Double precision.

__global__
void oskar_cudak_stokes_to_local_coherency_matrix_d(int ns, const double* ra,
        const double* dec, const double* stokes_I, const double* stokes_Q,
        const double* stokes_U, const double* stokes_V, double cos_lat,
        double sin_lat, double lst, double4c* coherency_matrix)
{
    // Get the source ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    // Copy the data from global memory.
    double c_ha, c_dec, c_I, c_Q, c_U, c_V;
    if (s < ns)
    {
        c_ha = ra[s]; // Source RA, but will be source hour angle.
        c_dec = dec[s];
        c_I = stokes_I[s];
        c_Q = stokes_Q[s];
        c_U = stokes_U[s];
        c_V = stokes_V[s];
    }
    __syncthreads();

    // Compute the source hour angle.
    c_ha = lst - c_ha; // HA = LST - RA.

    // Compute the source parallactic angle.
    double sin_dec, cos_dec, sin_a, cos_a;
    sincos(c_dec, &sin_dec, &cos_dec);
    sincos(c_ha, &sin_a, &cos_a);
    double y = cos_lat * sin_a;
    double x = sin_lat * cos_dec - cos_lat * sin_dec * cos_a;
    double q = 2.0 * atan2(y, x); // 2.0 * (parallactic angle)
    sincos(q, &sin_a, &cos_a);

    // Compute the source coherency matrix.
    x = -c_Q * cos_a - c_U * sin_a; // Q = -Q' cos(2q) - U' sin(2q)
    y = -c_U * cos_a + c_Q * sin_a; // U = -U' cos(2q) + Q' sin(2q)
    double4c b;
    b.a.x = c_I + x; // I + Q
    b.a.y = 0.0;
    b.b.x = y; // U + iV
    b.b.y = c_V;
    b.c.x = y; // U - iV
    b.c.y = -c_V;
    b.d.x = c_I - x; // I - Q
    b.d.y = 0.0;

    // Copy the source coherency matrix to global memory.
    __syncthreads();
    if (s < ns)
        coherency_matrix[s] = b;
}
