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

#include "sky/cudak/oskar_cudak_transform_to_local_stokes.h"
#include "sky/cudak/oskar_cudaf_parallactic_angle.h"

// Single precision.
__global__
void oskar_cudak_transform_to_local_stokes_f(int ns, const float* ra,
        const float* dec, float cos_lat, float sin_lat, float lst,
        float* stokes_Q, float* stokes_U)
{
    // Get the source ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    // Copy the data from global memory.
    float c_ha, c_dec, c_Q, c_U;
    if (s < ns)
    {
        c_ha = ra[s]; // Source RA, but will be source hour angle.
        c_dec = dec[s];
        c_Q = stokes_Q[s];
        c_U = stokes_U[s];
    }

    // Compute the source hour angle.
    c_ha = lst - c_ha; // HA = LST - RA.

    // Compute the source parallactic angle.
    float q = 2.0f * oskar_cudaf_parallactic_angle_f(c_ha, c_dec,
            cos_lat, sin_lat);
    float sin_a, cos_a;
    sincosf(q, &sin_a, &cos_a);

    // Compute the modified Stokes parameters.
    float l_Q = -c_Q * cos_a - c_U * sin_a; // Q = -Q' cos(2q) - U' sin(2q)
    float l_U = -c_U * cos_a + c_Q * sin_a; // U = -U' cos(2q) + Q' sin(2q)

    // Copy the modified Stokes parameters back to global memory.
    if (s < ns)
    {
        stokes_Q[s] = l_Q;
        stokes_U[s] = l_U;
    }
}

// Double precision.
__global__
void oskar_cudak_transform_to_local_stokes_d(int ns, const double* ra,
        const double* dec, double cos_lat, double sin_lat, double lst,
        double* stokes_Q, double* stokes_U)
{
    // Get the source ID that this thread is working on.
    const int s = blockDim.x * blockIdx.x + threadIdx.x;

    // Copy the data from global memory.
    double c_ha, c_dec, c_Q, c_U;
    if (s < ns)
    {
        c_ha = ra[s]; // Source RA, but will be source hour angle.
        c_dec = dec[s];
        c_Q = stokes_Q[s];
        c_U = stokes_U[s];
    }

    // Compute the source hour angle.
    c_ha = lst - c_ha; // HA = LST - RA.

    // Compute the source parallactic angle.
    double q = 2.0 * oskar_cudaf_parallactic_angle_d(c_ha, c_dec,
            cos_lat, sin_lat);
    double sin_a, cos_a;
    sincos(q, &sin_a, &cos_a);

    // Compute the modified Stokes parameters.
    double l_Q = -c_Q * cos_a - c_U * sin_a; // Q = -Q' cos(2q) - U' sin(2q)
    double l_U = -c_U * cos_a + c_Q * sin_a; // U = -U' cos(2q) + Q' sin(2q)

    // Copy the modified Stokes parameters back to global memory.
    if (s < ns)
    {
        stokes_Q[s] = l_Q;
        stokes_U[s] = l_U;
    }
}
