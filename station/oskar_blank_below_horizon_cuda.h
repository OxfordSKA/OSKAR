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

#ifndef OSKAR_BLANK_BELOW_HORIZON_CUDA_H_
#define OSKAR_BLANK_BELOW_HORIZON_CUDA_H_

/**
 * @file oskar_blank_below_horizon_cuda.h
 */

#include <oskar_global.h>
#include <oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Function to blank sources below the horizon using CUDA (single precision).
 *
 * @details
 * This CUDA function sets individual Jones scalars to zero for those
 * sources that are below the horizon.
 *
 * For sources where the mask value is negative, the corresponding element
 * of the Jones array is set to zero.
 *
 * Note that all pointers refer to device memory.
 *
 * @param[in,out] d_jones    Array of Jones scalars.
 * @param[in] num_sources    Number of source positions.
 * @param[in] d_mask         Array of source mask values.
 */
OSKAR_EXPORT
void oskar_blank_below_horizon_scalar_cuda_f(float2* d_jones,
        int num_sources, const float* d_mask);

/**
 * @brief
 * Function to blank sources below the horizon using CUDA (single precision).
 *
 * @details
 * This CUDA function sets individual Jones matrices to zero for those
 * sources that are below the horizon.
 *
 * For sources where the mask value is negative, the corresponding element
 * of the Jones array is set to zero.
 *
 * Note that all pointers refer to device memory.
 *
 * @param[in,out] d_jones    Array of Jones matrices.
 * @param[in] num_sources    Number of source positions.
 * @param[in] d_mask         Array of source mask values.
 */
OSKAR_EXPORT
void oskar_blank_below_horizon_matrix_cuda_f(float4c* d_jones,
        int num_sources, const float* d_mask);

/**
 * @brief
 * Function to blank sources below the horizon using CUDA (double precision).
 *
 * @details
 * This CUDA function sets individual Jones scalars to zero for those
 * sources that are below the horizon.
 *
 * For sources where the mask value is negative, the corresponding element
 * of the Jones array is set to zero.
 *
 * Note that all pointers refer to device memory.
 *
 * @param[in,out] d_jones    Array of Jones scalars.
 * @param[in] num_sources    Number of source positions.
 * @param[in] d_mask         Array of source mask values.
 */
OSKAR_EXPORT
void oskar_blank_below_horizon_scalar_cuda_d(double2* d_jones,
        int num_sources, const double* d_mask);

/**
 * @brief
 * Function to blank sources below the horizon using CUDA (double precision).
 *
 * @details
 * This CUDA function sets individual Jones matrices to zero for those
 * sources that are below the horizon.
 *
 * For sources where the mask value is negative, the corresponding element
 * of the Jones array is set to zero.
 *
 * Note that all pointers refer to device memory.
 *
 * @param[in,out] d_jones    Array of Jones matrices.
 * @param[in] num_sources    Number of source positions.
 * @param[in] d_mask         Array of source mask values.
 */
OSKAR_EXPORT
void oskar_blank_below_horizon_matrix_cuda_d(double4c* d_jones,
        int num_sources, const double* d_mask);

#ifdef __CUDACC__

/* Kernels. */

__global__
void oskar_blank_below_horizon_scalar_cudak_f(float2* jones,
        const int num_sources, const float* mask);

__global__
void oskar_blank_below_horizon_matrix_cudak_f(float4c* jones,
        const int num_sources, const float* mask);

__global__
void oskar_blank_below_horizon_scalar_cudak_d(double2* jones,
        const int num_sources, const double* mask);

__global__
void oskar_blank_below_horizon_matrix_cudak_d(double4c* jones,
        const int num_sources, const double* mask);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_BLANK_BELOW_HORIZON_CUDA_H_ */
