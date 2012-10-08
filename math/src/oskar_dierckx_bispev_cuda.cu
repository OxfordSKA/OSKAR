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

#include "math/oskar_dierckx_bispev_cuda.h"
#include "math/cudak/oskar_cudaf_dierckx_fpbisp_single.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Kernel wrappers. ======================================================== */

/* Single precision. */
void oskar_dierckx_bispev_cuda_f(const float* d_tx, int nx,
        const float* d_ty, int ny, const float* d_c, int kx, int ky,
        int n, const float* d_x, const float* d_y, int stride, float* d_z)
{
    /* Evaluate surface at the points by calling kernel. */
    int num_blocks, num_threads = 256;
    num_blocks = (n + num_threads - 1) / num_threads;
    oskar_dierckx_bispev_cudak_f
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (d_tx, nx, d_ty, ny, d_c,
            kx, ky, n, d_x, d_y, stride, d_z);
}

/* Double precision. */
void oskar_dierckx_bispev_cuda_d(const double* d_tx, int nx,
        const double* d_ty, int ny, const double* d_c, int kx, int ky,
        int n, const double* d_x, const double* d_y, int stride, double* d_z)
{
    /* Evaluate surface at the points by calling kernel. */
    int num_blocks, num_threads = 256;
    num_blocks = (n + num_threads - 1) / num_threads;
    oskar_dierckx_bispev_cudak_d
    OSKAR_CUDAK_CONF(num_blocks, num_threads) (d_tx, nx, d_ty, ny, d_c,
            kx, ky, n, d_x, d_y, stride, d_z);
}


/* Kernels. ================================================================ */

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

/* Double precision. */
__global__
void oskar_dierckx_bispev_cudak_d(const double* tx, const int nx,
        const double* ty, const int ny, const double* c, const int kx,
        const int ky, const int n, const double* x, const double* y,
        const int stride, double* z)
{
    // Get the output position (pixel) ID that this thread is working on. */
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;

    // Call device function to evaluate surface. */
    oskar_cudaf_dierckx_fpbisp_single_d(tx, nx, ty, ny, c, kx, ky,
            x[i], y[i], &z[i * stride]);
}

#ifdef __cplusplus
}
#endif
