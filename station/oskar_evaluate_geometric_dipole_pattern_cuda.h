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
#include <oskar_vector_types.h>

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
 * The output matrix is
 *
 * ( g_theta^a   g_phi^a )
 * ( g_theta^b   g_phi^b )
 *
 * where phi and theta are the angles measured from x to y and from xy to z,
 * respectively.
 *
 * The supplied theta and phi positions of the sources are the <b>modified</b>
 * source positions. They must be adjusted relative to a dipole with its axis
 * oriented along the x-direction.
 *
 * Note that all pointers refer to device memory.
 *
 * @param[in] num_points         Number of points.
 * @param[in] d_theta            Source position (modified) theta values in rad.
 * @param[in] d_phi              Source position (modified) phi values in rad.
 * @param[in] return_x_dipole    If true, return X dipole; else return Y dipole.
 * @param[out] d_pattern         Array of output Jones matrices per source.
 */
OSKAR_EXPORT
void oskar_evaluate_geometric_dipole_pattern_cuda_f(int num_points,
        const float* d_theta, const float* d_phi, int return_x_dipole,
        float4c* d_pattern);

/**
 * @brief
 * Evaluates pattern of a perfect dipole at source positions using CUDA
 * (double precision).
 *
 * @details
 * This function evaluates the pattern of a perfect dipole antenna
 * at the supplied source positions using CUDA.
 *
 * The output matrix is
 *
 * ( g_theta^a   g_phi^a )
 * ( g_theta^b   g_phi^b )
 *
 * where phi and theta are the angles measured from x to y and from xy to z,
 * respectively.
 *
 * The supplied theta and phi positions of the sources are the <b>modified</b>
 * source positions. They must be adjusted relative to a dipole with its axis
 * oriented along the x-direction.
 *
 * Note that all pointers refer to device memory.
 *
 * @param[in] num_points         Number of points.
 * @param[in] d_theta            Source position (modified) theta values in rad.
 * @param[in] d_phi              Source position (modified) phi values in rad.
 * @param[in] return_x_dipole    If true, return X dipole; else return Y dipole.
 * @param[out] d_pattern         Array of output Jones matrices per source.
 */
OSKAR_EXPORT
void oskar_evaluate_geometric_dipole_pattern_cuda_d(int num_points,
        const double* d_theta, const double* d_phi, int return_x_dipole,
        double4c* d_pattern);

#ifdef __CUDACC__

/**
 * @brief
 * Evaluates pattern of a perfect dipole at source positions (single precision).
 *
 * @details
 * This CUDA kernel evaluates the pattern of a perfect dipole antenna
 * at the supplied source positions.
 *
 * The output matrix is
 *
 * ( g_theta^a   g_phi^a )
 * ( g_theta^b   g_phi^b )
 *
 * where phi and theta are the angles measured from x to y and from xy to z,
 * respectively.
 *
 * The supplied theta and phi positions of the sources are the <b>modified</b>
 * source positions. They must be adjusted relative to a dipole with its axis
 * oriented along the x-direction.
 *
 * @param[in] num_points         Number of points.
 * @param[in] theta              Source position (modified) theta values in rad.
 * @param[in] phi                Source position (modified) phi values in rad.
 * @param[in] return_x_dipole    If true, return X dipole; else return Y dipole.
 * @param[out] pattern           Array of output Jones matrices per source.
 */
__global__
void oskar_evaluate_geometric_dipole_pattern_cudak_f(const int num_points,
        const float* theta, const float* phi, const int return_x_dipole,
        float4c* pattern);

/**
 * @brief
 * Evaluates pattern of a perfect dipole at source positions (double precision).
 *
 * @details
 * This CUDA kernel evaluates the pattern of a perfect dipole antenna
 * at the supplied source positions.
 *
 * The output matrix is
 *
 * ( g_theta^a   g_phi^a )
 * ( g_theta^b   g_phi^b )
 *
 * where phi and theta are the angles measured from x to y and from xy to z,
 * respectively.
 *
 * The supplied theta and phi positions of the sources are the <b>modified</b>
 * source positions. They must be adjusted relative to a dipole with its axis
 * oriented along the x-direction.
 *
 * @param[in] num_points         Number of points.
 * @param[in] theta              Source position (modified) theta values in rad.
 * @param[in] phi                Source position (modified) phi values in rad.
 * @param[in] return_x_dipole    If true, return X dipole; else return Y dipole.
 * @param[out] pattern           Array of output Jones matrices per source.
 */
__global__
void oskar_evaluate_geometric_dipole_pattern_cudak_d(const int num_points,
        const double* theta, const double* phi, const int return_x_dipole,
        double4c* pattern);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_GEOMETRIC_DIPOLE_PATTERN_CUDA_H_ */
