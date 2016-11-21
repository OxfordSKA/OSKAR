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

#ifndef OSKAR_DIERCKX_BISPEV_CUDA_H_
#define OSKAR_DIERCKX_BISPEV_CUDA_H_

/**
 * @file oskar_dierckx_bispev_cuda.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Function to evaluate B-spline coefficients using CUDA (single precision).
 *
 * @details
 * This function evaluates B-spline coefficients to determine
 * values of a fitted surface at the specified points using CUDA.
 *
 * Note that all pointers refer to device memory.
 *
 * @param[in] d_tx     Array of knot positions in x.
 * @param[in] nx       Number of knot positions in x.
 * @param[in] d_ty     Array of knot positions in y.
 * @param[in] ny       Number of knot positions in y.
 * @param[in] d_c      Array of spline coefficients.
 * @param[in] kx       Order of spline in x (use 3 for bicubic).
 * @param[in] ky       Order of spline in y (use 3 for bicubic).
 * @param[in] n        Number of points to evaluate.
 * @param[in] d_x      Input x positions.
 * @param[in] d_y      Input y positions.
 * @param[in] stride   Memory stride of output values (use 1 for contiguous).
 * @param[out] d_z     Output surface values.
 */
OSKAR_EXPORT
void oskar_dierckx_bispev_cuda_f(const float* d_tx, int nx,
        const float* d_ty, int ny, const float* d_c, int kx, int ky,
        int n, const float* d_x, const float* d_y, int stride, float* d_z);

/**
 * @brief
 * Function to evaluate B-spline coefficients using CUDA (double precision).
 *
 * @details
 * This function evaluates B-spline coefficients to determine
 * values of a fitted surface at the specified points using CUDA.
 *
 * Note that all pointers refer to device memory.
 *
 * @param[in] d_tx     Array of knot positions in x.
 * @param[in] nx       Number of knot positions in x.
 * @param[in] d_ty     Array of knot positions in y.
 * @param[in] ny       Number of knot positions in y.
 * @param[in] d_c      Array of spline coefficients.
 * @param[in] kx       Order of spline in x (use 3 for bicubic).
 * @param[in] ky       Order of spline in y (use 3 for bicubic).
 * @param[in] n        Number of points to evaluate.
 * @param[in] d_x      Input x positions.
 * @param[in] d_y      Input y positions.
 * @param[in] stride   Memory stride of output values (use 1 for contiguous).
 * @param[out] d_z     Output surface values.
 */
OSKAR_EXPORT
void oskar_dierckx_bispev_cuda_d(const double* d_tx, int nx,
        const double* d_ty, int ny, const double* d_c, int kx, int ky,
        int n, const double* d_x, const double* d_y, int stride, double* d_z);

#ifdef __CUDACC__

/* Kernels. */

__global__
void oskar_dierckx_bispev_cudak_f(const float* tx, const int nx,
        const float* ty, const int ny, const float* c, const int kx,
        const int ky, const int n, const float* x, const float* y,
        const int stride, float* z);

__global__
void oskar_dierckx_bispev_cudak_d(const double* tx, const int nx,
        const double* ty, const int ny, const double* c, const int kx,
        const int ky, const int n, const double* x, const double* y,
        const int stride, double* z);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_DIERCKX_BISPEV_CUDA_H_ */
