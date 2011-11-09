/*
 * Copyright (c) 2011, The University of Oxford
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

#ifndef OSKAR_CUDAK_BLANK_BELOW_HORIZON_H_
#define OSKAR_CUDAK_BLANK_BELOW_HORIZON_H_

/**
 * @file oskar_cudak_blank_below_horizon.h
 */

#include "oskar_global.h"
#include "utility/oskar_vector_types.h"

/**
 * @brief
 * CUDA kernel to blank sources below the horizon
 * (single precision).
 *
 * @details
 * This CUDA kernel sets individual Jones scalars to zero for those
 * sources that are below the horizon.
 *
 * For sources where the mask value is negative, the corresponding element
 * of the Jones array is set to zero.
 *
 * Note:
 * - One thread is assigned per source.
 * - Threads are assigned based on linear 'x' block and grid dimensions.
 *
 * @param[in] n         Number of source positions.
 * @param[in] mask      Array of source mask values.
 * @param[in,out] jones Array of Jones scalars.
 */
__global__
void oskar_cudak_blank_below_horizon_scalar_f(const int n, const float* mask,
        float2* jones);

/**
 * @brief
 * CUDA kernel to blank sources below the horizon
 * (single precision).
 *
 * @details
 * This CUDA kernel sets individual Jones matrices to zero for those
 * sources that are below the horizon.
 *
 * For sources where the mask value is negative, the corresponding element
 * of the Jones array is set to zero.
 *
 * Note:
 * - One thread is assigned per source.
 * - Threads are assigned based on linear 'x' block and grid dimensions.
 *
 * @param[in] n         Number of source positions.
 * @param[in] mask      Array of source mask values.
 * @param[in,out] jones Array of Jones matrices.
 */
__global__
void oskar_cudak_blank_below_horizon_matrix_f(const int n, const float* mask,
        float4c* jones);

/**
 * @brief
 * CUDA kernel to blank sources below the horizon
 * (double precision).
 *
 * @details
 * This CUDA kernel sets individual Jones scalars to zero for those
 * sources that are below the horizon.
 *
 * For sources where the mask value is negative, the corresponding element
 * of the Jones array is set to zero.
 *
 * @param[in] n         Number of source positions.
 * @param[in] mask      Array of source mask values.
 * @param[in,out] jones Array of Jones scalars.
 */
__global__
void oskar_cudak_blank_below_horizon_scalar_d(const int n, const double* mask,
        double2* jones);

/**
 * @brief
 * CUDA kernel to blank sources below the horizon
 * (double precision).
 *
 * @details
 * This CUDA kernel sets individual Jones matrices to zero for those
 * sources that are below the horizon.
 *
 * For sources where the mask value is negative, the corresponding element
 * of the Jones array is set to zero.
 *
 * @param[in] n         Number of source positions.
 * @param[in] mask      Array of source mask values.
 * @param[in,out] jones Array of Jones matrices.
 */
__global__
void oskar_cudak_blank_below_horizon_matrix_d(const int n, const double* mask,
        double4c* jones);

#endif // OSKAR_CUDAK_BLANK_BELOW_HORIZON_H_
