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

#include <oskar_dierckx_bispev_cuda.h>

#ifdef __cplusplus
extern "C" {
#endif

static __global__ void oskar_set_zeros_f(float* out, int n, int stride);
static __global__ void oskar_set_zeros_d(double* out, int n, int stride);

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_dierckx_bispev_cuda_f(const float* d_tx, int nx,
        const float* d_ty, int ny, const float* d_c, int kx, int ky,
        int n, const float* d_x, const float* d_y, int stride, float* d_z)
{
    /* Evaluate surface at the points by calling kernel. */
    int num_blocks, num_threads = 256;
    num_blocks = (n + num_threads - 1) / num_threads;
    if (!d_tx || !d_ty || !d_c || nx == 0 || ny == 0)
    {
        oskar_set_zeros_f
        OSKAR_CUDAK_CONF(num_blocks, num_threads) (d_z, n, stride);
    }
    else
    {
        oskar_dierckx_bispev_cudak_f
        OSKAR_CUDAK_CONF(num_blocks, num_threads) (d_tx, nx, d_ty, ny,
                d_c, kx, ky, n, d_x, d_y, stride, d_z);
    }
}

/* Double precision. */
void oskar_dierckx_bispev_cuda_d(const double* d_tx, int nx,
        const double* d_ty, int ny, const double* d_c, int kx, int ky,
        int n, const double* d_x, const double* d_y, int stride, double* d_z)
{
    /* Evaluate surface at the points by calling kernel. */
    int num_blocks, num_threads = 256;
    num_blocks = (n + num_threads - 1) / num_threads;
    if (!d_tx || !d_ty || !d_c || nx == 0 || ny == 0)
    {
        oskar_set_zeros_d
        OSKAR_CUDAK_CONF(num_blocks, num_threads) (d_z, n, stride);
    }
    else
    {
        oskar_dierckx_bispev_cudak_d
        OSKAR_CUDAK_CONF(num_blocks, num_threads) (d_tx, nx, d_ty, ny,
                d_c, kx, ky, n, d_x, d_y, stride, d_z);
    }
}


/* Kernels and device functions. ========================================== */

/**
 * @brief
 * CUDA device function for fpbspl from DIERCKX library (single precision).
 *
 * @details
 * CUDA device function to replace the fpbspl function from the DIERCKX
 * fitting library.
 *
 * This routine evaluates the (k+1) non-zero b-splines of degree k
 * at t(l) <= x < t(l+1) using the stable recurrence relation of
 * de Boor and Cox.
 */
__device__
void oskar_cudaf_dierckx_fpbspl_f(const float *t, const int k,
        const float x, const int l, float *h)
{
    float f, hh[5];
    int i, j, li, lj;

    h[0] = 1.0f;
    for (j = 1; j <= k; ++j)
    {
        for (i = 0; i < j; ++i)
        {
            hh[i] = h[i];
        }
        h[0] = 0.0f;
        for (i = 0; i < j; ++i)
        {
            li = l + i;
            lj = li - j;
            f = hh[i] / (t[li] - t[lj]);
            h[i] += f * (t[li] - x);
            h[i + 1] = f * (x - t[lj]);
        }
    }
}

/**
 * @brief
 * CUDA device function for fpbspl from DIERCKX library (double precision).
 *
 * @details
 * CUDA device function to replace the fpbspl function from the DIERCKX
 * fitting library.
 *
 * This routine evaluates the (k+1) non-zero b-splines of degree k
 * at t(l) <= x < t(l+1) using the stable recurrence relation of
 * de Boor and Cox.
 */
__device__
void oskar_cudaf_dierckx_fpbspl_d(const double *t, const int k,
        const double x, const int l, double *h)
{
    double f, hh[5];
    int i, j, li, lj;

    h[0] = 1.0;
    for (j = 1; j <= k; ++j)
    {
        for (i = 0; i < j; ++i)
        {
            hh[i] = h[i];
        }
        h[0] = 0.0;
        for (i = 0; i < j; ++i)
        {
            li = l + i;
            lj = li - j;
            f = hh[i] / (t[li] - t[lj]);
            h[i] += f * (t[li] - x);
            h[i + 1] = f * (x - t[lj]);
        }
    }
}

/**
 * @brief
 * CUDA device function for fpbisp from DIERCKX library (single precision).
 *
 * @details
 * CUDA device function to replace the fpbisp function from the DIERCKX
 * fitting library.
 */
__device__
void oskar_cudaf_dierckx_fpbisp_single_f(const float *tx, const int nx,
        const float *ty, const int ny, const float *c, const int kx,
        const int ky, float x, float y, float *z)
{
    int j, l, l1, l2, k1, nk1, lx;
    float wx[6], wy[6], t;

    /* Do x. */
    k1 = kx + 1;
    nk1 = nx - k1;
    t = tx[kx];
    if (x < t) x = t;
    t = tx[nk1];
    if (x > t) x = t;
    l = k1;
    while (!(x < tx[l] || l == nk1)) l++;
    oskar_cudaf_dierckx_fpbspl_f(tx, kx, x, l, wx);
    lx = l - k1;

    /* Do y. */
    k1 = ky + 1;
    nk1 = ny - k1;
    t = ty[ky];
    if (y < t) y = t;
    t = ty[nk1];
    if (y > t) y = t;
    l = k1;
    while (!(y < ty[l] || l == nk1)) l++;
    oskar_cudaf_dierckx_fpbspl_f(ty, ky, y, l, wy);
    l1 = lx * nk1 + (l - k1);

    /* Evaluate surface using coefficients. */
    t = 0.0f;
    for (l = 0; l <= kx; ++l)
    {
        l2 = l1;
        for (j = 0; j <= ky; ++j)
        {
            t += c[l2] * wx[l] * wy[j];
            ++l2;
        }
        l1 += nk1;
    }
    *z = t;
}

/**
 * @brief
 * CUDA device function for fpbisp from DIERCKX library (double precision).
 *
 * @details
 * CUDA device function to replace the fpbisp function from the DIERCKX
 * fitting library.
 */
__device__
void oskar_cudaf_dierckx_fpbisp_single_d(const double *tx, const int nx,
        const double *ty, const int ny, const double *c, const int kx,
        const int ky, double x, double y, double *z)
{
    int j, l, l1, l2, k1, nk1, lx;
    double wx[6], wy[6], t;

    /* Do x. */
    k1 = kx + 1;
    nk1 = nx - k1;
    t = tx[kx];
    if (x < t) x = t;
    t = tx[nk1];
    if (x > t) x = t;
    l = k1;
    while (!(x < tx[l] || l == nk1)) l++;
    oskar_cudaf_dierckx_fpbspl_d(tx, kx, x, l, wx);
    lx = l - k1;

    /* Do y. */
    k1 = ky + 1;
    nk1 = ny - k1;
    t = ty[ky];
    if (y < t) y = t;
    t = ty[nk1];
    if (y > t) y = t;
    l = k1;
    while (!(y < ty[l] || l == nk1)) l++;
    oskar_cudaf_dierckx_fpbspl_d(ty, ky, y, l, wy);
    l1 = lx * nk1 + (l - k1);

    /* Evaluate surface using coefficients. */
    t = 0.0;
    for (l = 0; l <= kx; ++l)
    {
        l2 = l1;
        for (j = 0; j <= ky; ++j)
        {
            t += c[l2] * wx[l] * wy[j];
            ++l2;
        }
        l1 += nk1;
    }
    *z = t;
}

/* Single precision. */
__global__
void oskar_dierckx_bispev_cudak_f(const float* tx, const int nx,
        const float* ty, const int ny, const float* c, const int kx,
        const int ky, const int n, const float* x, const float* y,
        const int stride, float* z)
{
    /* Get the output position (pixel) ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    /* Call device function to evaluate surface. */
    oskar_cudaf_dierckx_fpbisp_single_f(tx, nx, ty, ny, c, kx, ky,
            x[i], y[i], &z[i * stride]);
}

static __global__
void oskar_set_zeros_f(float* out, int n, int stride)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    out[i * stride] = 0.0f;
}

/* Double precision. */
__global__
void oskar_dierckx_bispev_cudak_d(const double* tx, const int nx,
        const double* ty, const int ny, const double* c, const int kx,
        const int ky, const int n, const double* x, const double* y,
        const int stride, double* z)
{
    /* Get the output position (pixel) ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    /* Call device function to evaluate surface. */
    oskar_cudaf_dierckx_fpbisp_single_d(tx, nx, ty, ny, c, kx, ky,
            x[i], y[i], &z[i * stride]);
}

static __global__
void oskar_set_zeros_d(double* out, int n, int stride)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    out[i * stride] = 0.0;
}

#ifdef __cplusplus
}
#endif
