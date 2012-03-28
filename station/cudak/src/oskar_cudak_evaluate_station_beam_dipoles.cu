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

#include "station/cudak/oskar_cudak_evaluate_station_beam_dipoles.h"
#include <math.h>

// Single precision.
// Value for max_in_chunk should be 448 in single precision.

// Shared memory pointer used by the kernel.
extern __shared__ float2 smem[];

__global__
void oskar_cudak_evaluate_station_beam_dipoles_f(const int num_antennas,
        const float* x, const float* y, const float* z,
        const float* cos_orientation_x, const float* sin_orientation_x,
        const float* cos_orientation_y, const float* sin_orientation_y,
        const float2* weights, const int num_sources, const float* l,
        const float* m, const float* n, const int max_in_chunk,
        float4c* pattern)
{
    // Source index being processed by the thread.
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    float2 e_phi_a = make_float2(0.0f, 0.0f);
    float2 e_theta_a = make_float2(0.0f, 0.0f);
    float2 e_phi_b = make_float2(0.0f, 0.0f);
    float2 e_theta_b = make_float2(0.0f, 0.0f);

    // Get source direction cosines.
    float ll = 0.0f, lm = 0.0f, ln = 0.0f;
    if (s < num_sources)
    {
        ll = l[s]; // Component along x-axis.
        lm = m[s]; // Component along y-axis.
        ln = n[s]; // Component along z-axis.
    }

    // Evaluate phi, the source (co-azimuth) angle from East (x) to North (y).
    float sin_phi, cos_phi; // Cannot use direction cosines here.
    sincosf(atan2f(lm, ll), &sin_phi, &cos_phi);

    // Evaluate unit vectors e_theta and e_phi at source position.
    // cos_theta = ln
    const float e_theta_x = ln * cos_phi; // Component of e_theta in x.
    const float e_theta_y = ln * sin_phi; // Component of e_theta in y.
    // e_phi_x = -sin_phi;
    // e_phi_y = cos_phi;

    // Initialise shared memory caches.
    float2* cw = smem; // Cached weights.
    float2* cp = cw + max_in_chunk; // Cached x,y positions.
    float2* cox = cp + max_in_chunk; // Cached orientation x data.
    float2* coy = cox + max_in_chunk; // Cached orientation y data.
    float* cz = (float*)(coy + max_in_chunk); // Cached z positions.

    // Cache a chunk of input positions and weights into shared memory.
    for (int start = 0; start < num_antennas; start += max_in_chunk)
    {
        int chunk_size = num_antennas - start;
        if (chunk_size > max_in_chunk)
            chunk_size = max_in_chunk;

        // There are blockDim.x threads available - need to copy
        // chunk_size pieces of data from global memory.
        for (int t = threadIdx.x; t < chunk_size; t += blockDim.x)
        {
            const int g = start + t; // Global input index.
            cw[t] = weights[g];
            cp[t].x = x[g];
            cp[t].y = y[g];
            cox[t].x = cos_orientation_x[g];
            cox[t].y = sin_orientation_x[g];
            coy[t].x = cos_orientation_y[g];
            coy[t].y = sin_orientation_y[g];
            cz[t] = z[g];
        }

        // Must synchronise before computing partial output for these inputs.
        __syncthreads();

        // Loop over input chunk.
        for (int i = 0; i < chunk_size; ++i)
        {
            // Calculate the phase for the output position.
            float2 signal, w = cw[i];
            float phase = ll * cp[i].x + lm * cp[i].y + ln * cz[i];
            sincosf(phase, &signal.y, &signal.x);

            // Dot products for dipole projection:
            // g_phi_a   = a_x * e_phi_x   + a_y * e_phi_y;
            // g_theta_a = a_x * e_theta_x + a_y * e_theta_y;
            // g_phi_b   = b_x * e_phi_x   + b_y * e_phi_y;
            // g_theta_b = b_x * e_theta_x + b_y * e_theta_y;
            float g_phi_a   = cox[i].y * -sin_phi  + cox[i].x * cos_phi;
            float g_theta_a = cox[i].y * e_theta_x + cox[i].x * e_theta_y;
            float g_phi_b   = coy[i].y * -sin_phi  + coy[i].x * cos_phi;
            float g_theta_b = coy[i].y * e_theta_x + coy[i].x * e_theta_y;

            // Perform complex multiply.
            float2 t;
            t.x = signal.x * w.x + signal.y * w.y;
            t.y = signal.x * w.y - signal.y * w.x;

            // Accumulate.
            e_phi_a.x   += t.x * g_phi_a;
            e_phi_a.y   += t.y * g_phi_a;
            e_theta_a.x += t.x * g_theta_a;
            e_theta_a.y += t.y * g_theta_a;
            e_phi_b.x   += t.x * g_phi_b;
            e_phi_b.y   += t.y * g_phi_b;
            e_theta_b.x += t.x * g_theta_b;
            e_theta_b.y += t.y * g_theta_b;
        }

        // Must synchronise again before loading in a new input chunk.
        __syncthreads();
    }

    // Store result.
    if (s < num_sources)
    {
        pattern[s].a = e_phi_a;
        pattern[s].b = e_theta_a;
        pattern[s].c = e_phi_b;
        pattern[s].d = e_theta_b;
    }
}

// Double precision.
// Value for max_in_chunk should be 224 in double precision.

// Shared memory pointer used by the kernel.
extern __shared__ double2 smemd[];

__global__
void oskar_cudak_evaluate_station_beam_dipoles_d(const int num_antennas,
        const double* x, const double* y, const double* z,
        const double* cos_orientation_x, const double* sin_orientation_x,
        const double* cos_orientation_y, const double* sin_orientation_y,
        const double2* weights, const int num_sources, const double* l,
        const double* m, const double* n, const int max_in_chunk,
        double4c* pattern)
{
    // Source index being processed by the thread.
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    double2 e_phi_a = make_double2(0.0f, 0.0f);
    double2 e_theta_a = make_double2(0.0f, 0.0f);
    double2 e_phi_b = make_double2(0.0f, 0.0f);
    double2 e_theta_b = make_double2(0.0f, 0.0f);

    // Get source direction cosines.
    double ll = 0.0, lm = 0.0, ln = 0.0;
    if (s < num_sources)
    {
        ll = l[s]; // Component along x-axis.
        lm = m[s]; // Component along y-axis.
        ln = n[s]; // Component along z-axis.
    }

    // Evaluate phi, the source (co-azimuth) angle from East (x) to North (y).
    double sin_phi, cos_phi; // Cannot use direction cosines here.
    sincos(atan2(lm, ll), &sin_phi, &cos_phi);

    // Evaluate unit vectors e_theta and e_phi at source position.
    // cos_theta = ln
    const double e_theta_x = ln * cos_phi; // Component of e_theta in x.
    const double e_theta_y = ln * sin_phi; // Component of e_theta in y.
    // e_phi_x = -sin_phi;
    // e_phi_y = cos_phi;

    // Initialise shared memory caches.
    double2* cw = smemd; // Cached weights.
    double2* cp = cw + max_in_chunk; // Cached x,y positions.
    double2* cox = cp + max_in_chunk; // Cached orientation x data.
    double2* coy = cox + max_in_chunk; // Cached orientation y data.
    double* cz = (double*)(coy + max_in_chunk); // Cached z positions.

    // Cache a chunk of input positions and weights into shared memory.
    for (int start = 0; start < num_antennas; start += max_in_chunk)
    {
        int chunk_size = num_antennas - start;
        if (chunk_size > max_in_chunk)
            chunk_size = max_in_chunk;

        // There are blockDim.x threads available - need to copy
        // chunk_size pieces of data from global memory.
        for (int t = threadIdx.x; t < chunk_size; t += blockDim.x)
        {
            const int g = start + t; // Global input index.
            cw[t] = weights[g];
            cp[t].x = x[g];
            cp[t].y = y[g];
            cox[t].x = cos_orientation_x[g];
            cox[t].y = sin_orientation_x[g];
            coy[t].x = cos_orientation_y[g];
            coy[t].y = sin_orientation_y[g];
            cz[t] = z[g];
        }

        // Must synchronise before computing partial output for these inputs.
        __syncthreads();

        // Loop over input chunk.
        for (int i = 0; i < chunk_size; ++i)
        {
            // Calculate the phase for the output position.
            double2 signal, w = cw[i];
            double phase = ll * cp[i].x + lm * cp[i].y + ln * cz[i];
            sincos(phase, &signal.y, &signal.x);

            // Dot products for dipole projection:
            // g_phi_a   = a_x * e_phi_x   + a_y * e_phi_y;
            // g_theta_a = a_x * e_theta_x + a_y * e_theta_y;
            // g_phi_b   = b_x * e_phi_x   + b_y * e_phi_y;
            // g_theta_b = b_x * e_theta_x + b_y * e_theta_y;
            double g_phi_a   = cox[i].y * -sin_phi  + cox[i].x * cos_phi;
            double g_theta_a = cox[i].y * e_theta_x + cox[i].x * e_theta_y;
            double g_phi_b   = coy[i].y * -sin_phi  + coy[i].x * cos_phi;
            double g_theta_b = coy[i].y * e_theta_x + coy[i].x * e_theta_y;

            // Perform complex multiply.
            double2 t;
            t.x = signal.x * w.x + signal.y * w.y;
            t.y = signal.x * w.y - signal.y * w.x;

            // Accumulate.
            e_phi_a.x   += t.x * g_phi_a;
            e_phi_a.y   += t.y * g_phi_a;
            e_theta_a.x += t.x * g_theta_a;
            e_theta_a.y += t.y * g_theta_a;
            e_phi_b.x   += t.x * g_phi_b;
            e_phi_b.y   += t.y * g_phi_b;
            e_theta_b.x += t.x * g_theta_b;
            e_theta_b.y += t.y * g_theta_b;
        }

        // Must synchronise again before loading in a new input chunk.
        __syncthreads();
    }

    // Store result.
    if (s < num_sources)
    {
        pattern[s].a = e_phi_a;
        pattern[s].b = e_theta_a;
        pattern[s].c = e_phi_b;
        pattern[s].d = e_theta_b;
    }
}
