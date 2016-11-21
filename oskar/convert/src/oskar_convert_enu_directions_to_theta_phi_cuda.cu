/*
 * Copyright (c) 2014, The University of Oxford
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

#include <oskar_convert_enu_directions_to_theta_phi_cuda.h>
#include <private_convert_enu_directions_to_theta_phi_inline.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_convert_enu_directions_to_theta_phi_cudak_f(
        const int num_points, const float* restrict x,
        const float* restrict y, const float* restrict z,
        const float delta_phi, float* restrict theta,
        float* restrict phi)
{
    /* Get the position ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_points) return;

    oskar_convert_enu_directions_to_theta_phi_inline_f(x[i], y[i], z[i],
            delta_phi, &theta[i], &phi[i]);
}

/* Double precision. */
__global__
void oskar_convert_enu_directions_to_theta_phi_cudak_d(
        const int num_points, const double* restrict x,
        const double* restrict y, const double* restrict z,
        const double delta_phi, double* restrict theta,
        double* restrict phi)
{
    /* Get the position ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_points) return;

    oskar_convert_enu_directions_to_theta_phi_inline_d(x[i], y[i], z[i],
            delta_phi, &theta[i], &phi[i]);
}


/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_convert_enu_directions_to_theta_phi_cuda_f(int num_points,
        const float* d_x, const float* d_y, const float* d_z,
        float delta_phi, float* d_theta, float* d_phi)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_points + num_threads - 1) / num_threads;
    oskar_convert_enu_directions_to_theta_phi_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, d_x, d_y, d_z,
            delta_phi, d_theta, d_phi);
}

/* Double precision. */
void oskar_convert_enu_directions_to_theta_phi_cuda_d(int num_points,
        const double* d_x, const double* d_y, const double* d_z,
        double delta_phi, double* d_theta, double* d_phi)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_points + num_threads - 1) / num_threads;
    oskar_convert_enu_directions_to_theta_phi_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, d_x, d_y, d_z,
            delta_phi, d_theta, d_phi);
}

#ifdef __cplusplus
}
#endif
