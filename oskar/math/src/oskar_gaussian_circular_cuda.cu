/*
 * Copyright (c) 2012-2017, The University of Oxford
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

#include "math/oskar_gaussian_circular_cuda.h"

/* Kernels. ================================================================ */

/* Single precision. */
__global__
void oskar_gaussian_circular_cudak_cf(const int n, const float* restrict x,
        const float* restrict y, const float inv_2_var, float2* restrict z)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const float x_ = x[i];
    const float y_ = y[i];
    const float arg = (x_*x_ + y_*y_) * inv_2_var;
    z[i].x = expf(-arg);
    z[i].y = 0.0f;
}

__global__
void oskar_gaussian_circular_cudak_mf(const int n, const float* restrict x,
        const float* restrict y, const float inv_2_var, float4c* restrict z)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const float x_ = x[i];
    const float y_ = y[i];
    const float arg = (x_*x_ + y_*y_) * inv_2_var;
    const float value = expf(-arg);
    z[i].a.x = value;
    z[i].a.y = 0.0f;
    z[i].b.x = 0.0f;
    z[i].b.y = 0.0f;
    z[i].c.x = 0.0f;
    z[i].c.y = 0.0f;
    z[i].d.x = value;
    z[i].d.y = 0.0f;
}


/* Double precision. */
__global__
void oskar_gaussian_circular_cudak_cd(const int n, const double* restrict x,
        const double* restrict y, const double inv_2_var, double2* restrict z)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const double x_ = x[i];
    const double y_ = y[i];
    const double arg = (x_*x_ + y_*y_) * inv_2_var;
    z[i].x = exp(-arg);
    z[i].y = 0.0;
}

__global__
void oskar_gaussian_circular_cudak_md(const int n, const double* restrict x,
        const double* restrict y, const double inv_2_var, double4c* restrict z)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    const double x_ = x[i];
    const double y_ = y[i];
    const double arg = (x_*x_ + y_*y_) * inv_2_var;
    const double value = exp(-arg);
    z[i].a.x = value;
    z[i].a.y = 0.0;
    z[i].b.x = 0.0;
    z[i].b.y = 0.0;
    z[i].c.x = 0.0;
    z[i].c.y = 0.0;
    z[i].d.x = value;
    z[i].d.y = 0.0;
}


#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_gaussian_circular_cuda_complex_f(int n, const float* d_x,
        const float* d_y, float std, float2* d_z)
{
    int num_blocks, num_threads = 256;
    float inv_2_var;
    num_blocks = (n + num_threads - 1) / num_threads;
    inv_2_var = 1.0f / (2.0f * std * std);
    oskar_gaussian_circular_cudak_cf OSKAR_CUDAK_CONF(num_blocks, num_threads)
    (n, d_x, d_y, inv_2_var, d_z);
}

void oskar_gaussian_circular_cuda_matrix_f(int n, const float* d_x,
        const float* d_y, float std, float4c* d_z)
{
    int num_blocks, num_threads = 256;
    float inv_2_var;
    num_blocks = (n + num_threads - 1) / num_threads;
    inv_2_var = 1.0f / (2.0f * std * std);
    oskar_gaussian_circular_cudak_mf OSKAR_CUDAK_CONF(num_blocks, num_threads)
    (n, d_x, d_y, inv_2_var, d_z);
}


/* Double precision. */
void oskar_gaussian_circular_cuda_complex_d(int n, const double* d_x,
        const double* d_y, double std, double2* d_z)
{
    int num_blocks, num_threads = 256;
    double inv_2_var;
    num_blocks = (n + num_threads - 1) / num_threads;
    inv_2_var = 1.0 / (2.0 * std * std);
    oskar_gaussian_circular_cudak_cd OSKAR_CUDAK_CONF(num_blocks, num_threads)
    (n, d_x, d_y, inv_2_var, d_z);
}

void oskar_gaussian_circular_cuda_matrix_d(int n, const double* d_x,
        const double* d_y, double std, double4c* d_z)
{
    int num_blocks, num_threads = 256;
    double inv_2_var;
    num_blocks = (n + num_threads - 1) / num_threads;
    inv_2_var = 1.0 / (2.0 * std * std);
    oskar_gaussian_circular_cudak_md OSKAR_CUDAK_CONF(num_blocks, num_threads)
    (n, d_x, d_y, inv_2_var, d_z);
}

#ifdef __cplusplus
}
#endif
