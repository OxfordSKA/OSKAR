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

#ifndef OSKAR_EVALUATE_BASELINES_CUDA_H_
#define OSKAR_EVALUATE_BASELINES_CUDA_H_

/**
 * @file oskar_evaluate_baselines_cuda.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * CUDA kernel wrapper to compute baseline coordinates for all station pairs
 * (single precision).
 *
 * @details
 * Given the (u,v,w) coordinates for each station, this CUDA function computes
 * the baseline coordinates for all station pairs.
 *
 * The output arrays must be pre-sized to length N * (N - 1) / 2, where N is
 * the number of stations.
 *
 * @param[out] d_uu           The baseline u-positions.
 * @param[out] d_vv           The baseline v-positions.
 * @param[out] d_ww           The baseline w-positions.
 * @param[in]  num_stations   The number of stations.
 * @param[in]  d_u            The station u-positions.
 * @param[in]  d_v            The station v-positions.
 * @param[in]  d_w            The station w-positions.
 */
OSKAR_EXPORT
void oskar_evaluate_baselines_cuda_f(float* d_uu, float* d_vv, float* d_ww,
        int num_stations, const float* d_u, const float* d_v,
        const float* d_w);

/**
 * @brief
 * CUDA kernel wrapper to compute baseline coordinates for all station pairs
 * (double precision).
 *
 * @details
 * Given the (u,v,w) coordinates for each station, this CUDA function computes
 * the baseline coordinates for all station pairs.
 *
 * The output arrays must be pre-sized to length N * (N - 1) / 2, where N is
 * the number of stations.
 *
 * @param[out] d_uu           The baseline u-positions.
 * @param[out] d_vv           The baseline v-positions.
 * @param[out] d_ww           The baseline w-positions.
 * @param[in]  num_stations   The number of stations.
 * @param[in]  d_u            The station u-positions.
 * @param[in]  d_v            The station v-positions.
 * @param[in]  d_w            The station w-positions.
 */
OSKAR_EXPORT
void oskar_evaluate_baselines_cuda_d(double* d_uu, double* d_vv, double* d_ww,
        int num_stations, const double* d_u, const double* d_v,
        const double* d_w);

#ifdef __CUDACC__

/**
 * @brief
 * CUDA kernel to compute baseline coordinates for all station pairs
 * (single precision).
 *
 * @details
 * Given the (u,v,w) coordinates for each station, this CUDA kernel computes
 * the baseline coordinates for all station pairs.
 *
 * The output arrays must be pre-sized to length N * (N - 1) / 2, where N is
 * the number of stations.
 *
 * @param[out] uu           The baseline u-positions.
 * @param[out] vv           The baseline v-positions.
 * @param[out] ww           The baseline w-positions.
 * @param[in]  num_stations The number of stations.
 * @param[in]  u            The station u-positions.
 * @param[in]  v            The station v-positions.
 * @param[in]  w            The station w-positions.
 */
__global__
void oskar_evaluate_baselines_cudak_f(float* uu, float* vv, float* ww,
        int num_stations, const float* u, const float* v, const float* w);

/**
 * @brief
 * CUDA kernel to compute baseline coordinates for all station pairs
 * (double precision).
 *
 * @details
 * Given the (u,v,w) coordinates for each station, this CUDA kernel computes
 * the baseline coordinates for all station pairs.
 *
 * The output arrays must be pre-sized to length N * (N - 1) / 2, where N is
 * the number of stations.
 *
 * @param[out] uu           The baseline u-positions.
 * @param[out] vv           The baseline v-positions.
 * @param[out] ww           The baseline w-positions.
 * @param[in]  num_stations The number of stations.
 * @param[in]  u            The station u-positions.
 * @param[in]  v            The station v-positions.
 * @param[in]  w            The station w-positions.
 */
__global__
void oskar_evaluate_baselines_cudak_d(double* uu, double* vv, double* ww,
        int num_stations, const double* u, const double* v, const double* w);

#endif /* __CUDACC__ */

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_BASELINES_CUDA_H_ */
