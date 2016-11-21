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

#include <oskar_convert_ludwig3_to_theta_phi_components_cuda.h>
#include <private_convert_ludwig3_to_theta_phi_components_inline.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_convert_ludwig3_to_theta_phi_components_cudak_f(
        const int num_points, float2* h_theta, float2* v_phi,
        const float* __restrict__ phi, const int stride)
{
    /* Get the position ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_points) return;

    oskar_convert_ludwig3_to_theta_phi_components_inline_f(&h_theta[i*stride],
            &v_phi[i*stride], phi[i]);
}

/* Double precision. */
__global__
void oskar_convert_ludwig3_to_theta_phi_components_cudak_d(
        const int num_points, double2* h_theta, double2* v_phi,
        const double* __restrict__ phi, const int stride)
{
    /* Get the position ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= num_points) return;

    oskar_convert_ludwig3_to_theta_phi_components_inline_d(&h_theta[i*stride],
            &v_phi[i*stride], phi[i]);
}


/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_convert_ludwig3_to_theta_phi_components_cuda_f(int num_points,
        float2* d_h_theta, float2* d_v_phi, const float* d_phi, int stride)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_points + num_threads - 1) / num_threads;
    oskar_convert_ludwig3_to_theta_phi_components_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, d_h_theta, d_v_phi,
            d_phi, stride);
}

/* Double precision. */
void oskar_convert_ludwig3_to_theta_phi_components_cuda_d(int num_points,
        double2* d_h_theta, double2* d_v_phi, const double* d_phi, int stride)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_points + num_threads - 1) / num_threads;
    oskar_convert_ludwig3_to_theta_phi_components_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, d_h_theta, d_v_phi,
            d_phi, stride);
}

#ifdef __cplusplus
}
#endif
