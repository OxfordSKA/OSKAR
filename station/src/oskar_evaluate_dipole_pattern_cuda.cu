/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <oskar_mem.h>
#include "station/oskar_evaluate_dipole_pattern_cuda.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_evaluate_dipole_pattern_cuda_f(int num_points,
        const float* d_theta, const float* d_phi, int return_x_dipole,
        float4c* d_pattern)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_points + num_threads - 1) / num_threads;
    oskar_evaluate_dipole_pattern_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, d_theta, d_phi,
            return_x_dipole, d_pattern);
}

/* Double precision. */
void oskar_evaluate_dipole_pattern_cuda_d(int num_points,
        const double* d_theta, const double* d_phi, int return_x_dipole,
        double4c* d_pattern)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_points + num_threads - 1) / num_threads;
    oskar_evaluate_dipole_pattern_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, d_theta, d_phi,
            return_x_dipole, d_pattern);
}


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_evaluate_dipole_pattern_cudak_f(const int num_points,
        const float* theta, const float* phi, const int return_x_dipole,
        float4c* pattern)
{
    float theta_c, sin_phi, cos_phi, cos_theta, e_theta, e_phi;

    /* Source index being processed by the thread. */
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_points) return;

    /* Get (modified) source vector components, relative to a dipole along x. */
    theta_c = theta[s];
    cos_theta = cosf(theta_c);
    sincosf(phi[s], &sin_phi, &cos_phi);

    /* Evaluate vectors e_theta and e_phi in x-direction at source position. */
    e_theta = cos_theta * cos_phi;
    e_phi = -sin_phi;

    /* Store components. */
    if (return_x_dipole)
    {
        pattern[s].a.x = e_theta;
        pattern[s].a.y = 0.0f;
        pattern[s].b.x = e_phi;
        pattern[s].b.y = 0.0f;
    }
    else
    {
        pattern[s].c.x = e_theta;
        pattern[s].c.y = 0.0f;
        pattern[s].d.x = e_phi;
        pattern[s].d.y = 0.0f;
    }
}

/* Double precision. */
__global__
void oskar_evaluate_dipole_pattern_cudak_d(const int num_points,
        const double* theta, const double* phi, const int return_x_dipole,
        double4c* pattern)
{
    double theta_c, sin_phi, cos_phi, cos_theta, e_theta, e_phi;

    /* Source index being processed by the thread. */
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_points) return;

    /* Get (modified) source vector components, relative to a dipole along x. */
    theta_c = theta[s];
    cos_theta = cos(theta_c);
    sincos(phi[s], &sin_phi, &cos_phi);

    /* Evaluate vectors e_theta and e_phi in x-direction at source position. */
    e_theta = cos_theta * cos_phi;
    e_phi = -sin_phi;

    /* Store components. */
    if (return_x_dipole)
    {
        pattern[s].a.x = e_theta;
        pattern[s].a.y = 0.0;
        pattern[s].b.x = e_phi;
        pattern[s].b.y = 0.0;
    }
    else
    {
        pattern[s].c.x = e_theta;
        pattern[s].c.y = 0.0;
        pattern[s].d.x = e_phi;
        pattern[s].d.y = 0.0;
    }
}

#ifdef __cplusplus
}
#endif
