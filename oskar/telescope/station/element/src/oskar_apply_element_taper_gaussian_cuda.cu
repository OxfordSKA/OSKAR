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

#include "telescope/station/element/oskar_apply_element_taper_gaussian_cuda.h"

#define M_4LN2f 2.77258872223978123767f
#define M_4LN2  2.77258872223978123767

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

void oskar_apply_element_taper_gaussian_scalar_cuda_f(float2* d_jones,
        int num_sources, float fwhm, const float* d_theta)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_apply_element_taper_gaussian_scalar_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (d_jones, num_sources, fwhm,
            d_theta);
}

void oskar_apply_element_taper_gaussian_matrix_cuda_f(float4c* d_jones,
        int num_sources, float fwhm, const float* d_theta)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_apply_element_taper_gaussian_matrix_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (d_jones, num_sources, fwhm,
            d_theta);
}

void oskar_apply_element_taper_gaussian_scalar_cuda_d(double2* d_jones,
        int num_sources, double fwhm, const double* d_theta)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_apply_element_taper_gaussian_scalar_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (d_jones, num_sources, fwhm,
            d_theta);
}

void oskar_apply_element_taper_gaussian_matrix_cuda_d(double4c* d_jones,
        int num_sources, double fwhm, const double* d_theta)
{
    int num_blocks, num_threads = 256;
    num_blocks = (num_sources + num_threads - 1) / num_threads;
    oskar_apply_element_taper_gaussian_matrix_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (d_jones, num_sources, fwhm,
            d_theta);
}


/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_apply_element_taper_gaussian_scalar_cudak_f(float2* jones,
        const int num_sources, const float fwhm, const float* theta)
{
    float factor, theta_sq, inv_2sigma_sq;

    /* Source index being processed by thread. */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sources) return;

    /* Compute and apply tapering factor. */
    theta_sq = theta[i];
    theta_sq *= theta_sq;
    inv_2sigma_sq = M_4LN2f / (fwhm * fwhm);
    factor = expf(-theta_sq * inv_2sigma_sq);
    jones[i].x *= factor;
    jones[i].y *= factor;
}

__global__
void oskar_apply_element_taper_gaussian_matrix_cudak_f(float4c* jones,
        const int num_sources, const float fwhm, const float* theta)
{
    float factor, theta_sq, inv_2sigma_sq;

    /* Source index being processed by thread. */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sources) return;

    /* Compute and apply tapering factor. */
    theta_sq = theta[i];
    theta_sq *= theta_sq;
    inv_2sigma_sq = M_4LN2f / (fwhm * fwhm);
    factor = expf(-theta_sq * inv_2sigma_sq);
    jones[i].a.x *= factor;
    jones[i].a.y *= factor;
    jones[i].b.x *= factor;
    jones[i].b.y *= factor;
    jones[i].c.x *= factor;
    jones[i].c.y *= factor;
    jones[i].d.x *= factor;
    jones[i].d.y *= factor;
}

/* Double precision. */
__global__
void oskar_apply_element_taper_gaussian_scalar_cudak_d(double2* jones,
        const int num_sources, const double fwhm, const double* theta)
{
    double factor, theta_sq, inv_2sigma_sq;

    /* Source index being processed by thread. */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sources) return;

    /* Compute and apply tapering factor. */
    theta_sq = theta[i];
    theta_sq *= theta_sq;
    inv_2sigma_sq = M_4LN2 / (fwhm * fwhm);
    factor = exp(-theta_sq * inv_2sigma_sq);
    jones[i].x *= factor;
    jones[i].y *= factor;
}

__global__
void oskar_apply_element_taper_gaussian_matrix_cudak_d(double4c* jones,
        const int num_sources, const double fwhm, const double* theta)
{
    double factor, theta_sq, inv_2sigma_sq;

    /* Source index being processed by thread. */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_sources) return;

    /* Compute and apply tapering factor. */
    theta_sq = theta[i];
    theta_sq *= theta_sq;
    inv_2sigma_sq = M_4LN2 / (fwhm * fwhm);
    factor = exp(-theta_sq * inv_2sigma_sq);
    jones[i].a.x *= factor;
    jones[i].a.y *= factor;
    jones[i].b.x *= factor;
    jones[i].b.y *= factor;
    jones[i].c.x *= factor;
    jones[i].c.y *= factor;
    jones[i].d.x *= factor;
    jones[i].d.y *= factor;
}

#ifdef __cplusplus
}
#endif
