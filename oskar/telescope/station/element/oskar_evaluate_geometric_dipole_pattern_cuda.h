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

#ifndef OSKAR_EVALUATE_GEOMETRIC_DIPOLE_PATTERN_CUDA_H_
#define OSKAR_EVALUATE_GEOMETRIC_DIPOLE_PATTERN_CUDA_H_

/**
 * @file oskar_evaluate_geometric_dipole_pattern_cuda.h
 */

#include <oskar_global.h>
#include <utility/oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates pattern of a perfect dipole at source positions using CUDA
 * (single precision).
 *
 * @details
 * This function evaluates the pattern of a perfect dipole antenna
 * at the supplied source positions using CUDA.
 *
 * The supplied theta and phi positions of the sources are the <b>modified</b>
 * source positions. They must be adjusted relative to a dipole with its axis
 * oriented along the x-direction.
 *
 * Note that all pointers refer to device memory.
 *
 * @param[in] num_points         Number of points.
 * @param[in] d_theta            Point position (modified) theta values in rad.
 * @param[in] d_phi              Point position (modified) phi values in rad.
 * @param[in] stride             Stride into output arrays.
 * @param[out] d_E_theta         Response per point in E_theta.
 * @param[out] d_E_phi           Response per point in E_phi.
 */
OSKAR_EXPORT
void oskar_evaluate_geometric_dipole_pattern_cuda_f(int num_points,
        const float* d_theta, const float* d_phi, int stride,
        float2* d_E_theta, float2* d_E_phi);

/**
 * @brief
 * Evaluates pattern of a perfect dipole at source positions using CUDA
 * (scalar version, single precision).
 *
 * @details
 * This function evaluates the scalar pattern of a perfect dipole antenna
 * at the supplied source positions using CUDA.
 *
 * The supplied theta and phi positions of the sources are the <b>modified</b>
 * source positions. They must be adjusted relative to a dipole with its axis
 * oriented along the x-direction.
 *
 * Note that all pointers refer to device memory.
 *
 * @param[in] num_points         Number of points.
 * @param[in] d_theta            Point position (modified) theta values in rad.
 * @param[in] d_phi              Point position (modified) phi values in rad.
 * @param[in] stride             Stride into output array (normally 1).
 * @param[out] d_pattern         Response per point.
 */
OSKAR_EXPORT
void oskar_evaluate_geometric_dipole_pattern_scalar_cuda_f(int num_points,
        const float* d_theta, const float* d_phi, int stride,
        float2* d_pattern);

/**
 * @brief
 * Evaluates pattern of a perfect dipole at source positions using CUDA
 * (double precision).
 *
 * @details
 * This function evaluates the pattern of a perfect dipole antenna
 * at the supplied source positions using CUDA.
 *
 * The supplied theta and phi positions of the sources are the <b>modified</b>
 * source positions. They must be adjusted relative to a dipole with its axis
 * oriented along the x-direction.
 *
 * Note that all pointers refer to device memory.
 *
 * @param[in] num_points         Number of points.
 * @param[in] d_theta            Point position (modified) theta values in rad.
 * @param[in] d_phi              Point position (modified) phi values in rad.
 * @param[in] stride             Stride into output arrays.
 * @param[out] d_E_theta         Response per point in E_theta.
 * @param[out] d_E_phi           Response per point in E_phi.
 */
OSKAR_EXPORT
void oskar_evaluate_geometric_dipole_pattern_cuda_d(int num_points,
        const double* d_theta, const double* d_phi, int stride,
        double2* d_E_theta, double2* d_E_phi);

/**
 * @brief
 * Evaluates pattern of a perfect dipole at source positions using CUDA
 * (scalar version, double precision).
 *
 * @details
 * This function evaluates the scalar pattern of a perfect dipole antenna
 * at the supplied source positions using CUDA.
 *
 * The supplied theta and phi positions of the sources are the <b>modified</b>
 * source positions. They must be adjusted relative to a dipole with its axis
 * oriented along the x-direction.
 *
 * Note that all pointers refer to device memory.
 *
 * @param[in] num_points         Number of points.
 * @param[in] d_theta            Point position (modified) theta values in rad.
 * @param[in] d_phi              Point position (modified) phi values in rad.
 * @param[in] stride             Stride into output array (normally 1).
 * @param[out] d_pattern         Response per point.
 */
OSKAR_EXPORT
void oskar_evaluate_geometric_dipole_pattern_scalar_cuda_d(int num_points,
        const double* d_theta, const double* d_phi, int stride,
        double2* d_pattern);

#ifdef __CUDACC__

/* Kernels. */

__global__
void oskar_evaluate_geometric_dipole_pattern_scalar_cudak_f(
        const int num_points, const float* restrict theta,
        const float* restrict phi, const int stride,
        float2* restrict pattern);

__global__
void oskar_evaluate_geometric_dipole_pattern_cudak_f(const int num_points,
        const float* restrict theta, const float* restrict phi,
        const int stride, float2* E_theta, float2* E_phi);

__global__
void oskar_evaluate_geometric_dipole_pattern_cudak_d(const int num_points,
        const double* restrict theta, const double* restrict phi,
        const int stride, double2* E_theta, double2* E_phi);

__global__
void oskar_evaluate_geometric_dipole_pattern_scalar_cudak_d(
        const int num_points, const double* restrict theta,
        const double* restrict phi, const int stride,
        double2* restrict pattern);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_GEOMETRIC_DIPOLE_PATTERN_CUDA_H_ */
