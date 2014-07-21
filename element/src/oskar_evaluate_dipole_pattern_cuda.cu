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

#include <oskar_evaluate_dipole_pattern_cuda.h>
#include <oskar_evaluate_dipole_pattern_inline.h>
#include <oskar_cmath.h>

#ifdef __cplusplus
extern "C" {
#endif

#define C_0 299792458.0

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_evaluate_dipole_pattern_cuda_f(int num_points,
        const float* d_theta, const float* d_phi, float freq_hz,
        float dipole_length_m, int stride,
        float2* d_E_theta, float2* d_E_phi)
{
    int num_blocks, num_threads = 256;
    float kL, cos_kL;

    /* Precompute constants. */
    kL = dipole_length_m * (M_PI * freq_hz / C_0);
    cos_kL = (float)cos(kL);

    /* Call the kernel. */
    num_blocks = (num_points + num_threads - 1) / num_threads;
    oskar_evaluate_dipole_pattern_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, d_theta, d_phi,
            kL, cos_kL, stride, d_E_theta, d_E_phi);
}

void oskar_evaluate_dipole_pattern_scalar_cuda_f(int num_points,
        const float* d_theta, const float* d_phi, float freq_hz,
        float dipole_length_m, int stride, float2* d_pattern)
{
    int num_blocks, num_threads = 256;
    float kL, cos_kL;

    /* Precompute constants. */
    kL = dipole_length_m * (M_PI * freq_hz / C_0);
    cos_kL = (float)cos(kL);

    /* Call the kernel. */
    num_blocks = (num_points + num_threads - 1) / num_threads;
    oskar_evaluate_dipole_pattern_scalar_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, d_theta, d_phi,
            kL, cos_kL, stride, d_pattern);
}

/* Double precision. */
void oskar_evaluate_dipole_pattern_cuda_d(int num_points,
        const double* d_theta, const double* d_phi, double freq_hz,
        double dipole_length_m, int stride,
        double2* d_E_theta, double2* d_E_phi)
{
    int num_blocks, num_threads = 256;
    double kL, cos_kL;

    /* Precompute constants. */
    kL = dipole_length_m * (M_PI * freq_hz / C_0);
    cos_kL = cos(kL);

    /* Call the kernel. */
    num_blocks = (num_points + num_threads - 1) / num_threads;
    oskar_evaluate_dipole_pattern_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, d_theta, d_phi,
            kL, cos_kL, stride, d_E_theta, d_E_phi);
}

void oskar_evaluate_dipole_pattern_scalar_cuda_d(int num_points,
        const double* d_theta, const double* d_phi, double freq_hz,
        double dipole_length_m, int stride, double2* d_pattern)
{
    int num_blocks, num_threads = 256;
    double kL, cos_kL;

    /* Precompute constants. */
    kL = dipole_length_m * (M_PI * freq_hz / C_0);
    cos_kL = cos(kL);

    /* Call the kernel. */
    num_blocks = (num_points + num_threads - 1) / num_threads;
    oskar_evaluate_dipole_pattern_scalar_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (num_points, d_theta, d_phi,
            kL, cos_kL, stride, d_pattern);
}


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_evaluate_dipole_pattern_cudak_f(const int num_points,
        const float* restrict theta, const float* restrict phi,
        const float kL, const float cos_kL, const int stride,
        float2* E_theta, float2* E_phi)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_out = i * stride;
    if (i >= num_points) return;
    oskar_evaluate_dipole_pattern_inline_f(theta[i], phi[i], kL, cos_kL,
            E_theta + i_out, E_phi + i_out);
}

__global__
void oskar_evaluate_dipole_pattern_scalar_cudak_f(const int num_points,
        const float* restrict theta, const float* restrict phi,
        const float kL, const float cos_kL, const int stride,
        float2* restrict pattern)
{
    float theta_, phi_, amp;
    float4c val;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_out = i * stride;
    if (i >= num_points) return;

    /* Get source coordinates. */
    theta_ = theta[i];
    phi_ = phi[i];

    /* Evaluate E_theta, E_phi for both X and Y dipoles. */
    oskar_evaluate_dipole_pattern_inline_f(theta_,
            phi_, kL, cos_kL, &val.a, &val.b);
    oskar_evaluate_dipole_pattern_inline_f(theta_,
            phi_ + (float)M_PI_2, kL, cos_kL, &val.c, &val.d);

    /* Get sum of the diagonal of the autocorrelation matrix. */
    amp = val.a.x * val.a.x + val.a.y * val.a.y +
            val.b.x * val.b.x + val.b.y * val.b.y +
            val.c.x * val.c.x + val.c.y * val.c.y +
            val.d.x * val.d.x + val.d.y * val.d.y;
    amp = sqrtf(0.5f * amp);

    /* Save amplitude. */
    pattern[i_out].x = amp;
    pattern[i_out].y = 0.0f;
}

/* Double precision. */
__global__
void oskar_evaluate_dipole_pattern_cudak_d(const int num_points,
        const double* restrict theta, const double* restrict phi,
        const double kL, const double cos_kL, const int stride,
        double2* E_theta, double2* E_phi)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_out = i * stride;
    if (i >= num_points) return;
    oskar_evaluate_dipole_pattern_inline_d(theta[i], phi[i], kL, cos_kL,
            E_theta + i_out, E_phi + i_out);
}

__global__
void oskar_evaluate_dipole_pattern_scalar_cudak_d(const int num_points,
        const double* restrict theta, const double* restrict phi,
        const double kL, const double cos_kL, const int stride,
        double2* restrict pattern)
{
    double theta_, phi_, amp;
    double4c val;

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int i_out = i * stride;
    if (i >= num_points) return;

    /* Get source coordinates. */
    theta_ = theta[i];
    phi_ = phi[i];

    /* Evaluate E_theta, E_phi for both X and Y dipoles. */
    oskar_evaluate_dipole_pattern_inline_d(theta_,
            phi_, kL, cos_kL, &val.a, &val.b);
    oskar_evaluate_dipole_pattern_inline_d(theta_,
            phi_ + M_PI_2, kL, cos_kL, &val.c, &val.d);

    /* Get sum of the diagonal of the autocorrelation matrix. */
    amp = val.a.x * val.a.x + val.a.y * val.a.y +
            val.b.x * val.b.x + val.b.y * val.b.y +
            val.c.x * val.c.x + val.c.y * val.c.y +
            val.d.x * val.d.x + val.d.y * val.d.y;
    amp = sqrt(0.5 * amp);

    /* Save amplitude. */
    pattern[i_out].x = amp;
    pattern[i_out].y = 0.0;
}

#ifdef __cplusplus
}
#endif
