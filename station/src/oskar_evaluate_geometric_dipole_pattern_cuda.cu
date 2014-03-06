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

#include <oskar_evaluate_geometric_dipole_pattern_cuda.h>
#include <oskar_evaluate_geometric_dipole_pattern_inline.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_evaluate_geometric_dipole_pattern_cuda_f(int num_points,
        const float* d_theta, const float* d_phi, int return_x_dipole,
        float4c* d_pattern)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_points + num_threads - 1) / num_threads;
    oskar_evaluate_geometric_dipole_pattern_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, d_theta, d_phi,
            return_x_dipole, d_pattern);
}

/* Double precision. */
void oskar_evaluate_geometric_dipole_pattern_cuda_d(int num_points,
        const double* d_theta, const double* d_phi, int return_x_dipole,
        double4c* d_pattern)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_points + num_threads - 1) / num_threads;
    oskar_evaluate_geometric_dipole_pattern_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, d_theta, d_phi,
            return_x_dipole, d_pattern);
}


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_evaluate_geometric_dipole_pattern_cudak_f(const int num_points,
        const float* theta, const float* phi, const int return_x_dipole,
        float4c* pattern)
{
    float2 *E_theta, *E_phi;

    /* Source index being processed by the thread. */
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_points) return;

    /* Select the right outputs. */
    if (return_x_dipole)
    {
        E_theta = &(pattern[s].a);
        E_phi   = &(pattern[s].b);
    }
    else
    {
        E_theta = &(pattern[s].c);
        E_phi   = &(pattern[s].d);
    }
    oskar_evaluate_geometric_dipole_pattern_inline_f(theta[s], phi[s],
            E_theta, E_phi);

}

/* Double precision. */
__global__
void oskar_evaluate_geometric_dipole_pattern_cudak_d(const int num_points,
        const double* theta, const double* phi, const int return_x_dipole,
        double4c* pattern)
{
    double2 *E_theta, *E_phi;

    /* Source index being processed by the thread. */
    const int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= num_points) return;

    /* Select the right outputs. */
    if (return_x_dipole)
    {
        E_theta = &(pattern[s].a);
        E_phi   = &(pattern[s].b);
    }
    else
    {
        E_theta = &(pattern[s].c);
        E_phi   = &(pattern[s].d);
    }
    oskar_evaluate_geometric_dipole_pattern_inline_d(theta[s], phi[s],
            E_theta, E_phi);
}

#ifdef __cplusplus
}
#endif
