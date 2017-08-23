/*
 * Copyright (c) 2011-2017, The University of Oxford
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

#ifndef OSKAR_UPDATE_HORIZON_MASK_CUDA_H_
#define OSKAR_UPDATE_HORIZON_MASK_CUDA_H_

/**
 * @file oskar_update_horizon_mask_cuda.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Ensures source mask value is 1 if the source is visible
 * (single precision).
 *
 * @details
 * This kernel updates the horizon mask to determine whether a source is
 * visible from a particular station.
 *
 * @param[in] num_sources The number of source positions.
 * @param[in] d_l         Source l-direction cosines relative to phase centre.
 * @param[in] d_m         Source m-direction cosines relative to phase centre.
 * @param[in] d_n         Source n-direction cosines relative to phase centre.
 * @param[in] l_mul       Factor by which to multiply l-direction cosine.
 * @param[in] m_mul       Factor by which to multiply m-direction cosine.
 * @param[in] n_mul       Factor by which to multiply n-direction cosine.
 * @param[in,out] d_mask  The input and output mask vector.
 */
OSKAR_EXPORT
void oskar_update_horizon_mask_cuda_f(int num_sources, const float* d_l,
        const float* d_m, const float* d_n, const float l_mul,
        const float m_mul, const float n_mul, int* d_mask);

/**
 * @brief
 * Ensures source mask value is 1 if the source is visible
 * (double precision).
 *
 * @details
 * This kernel updates the horizon mask to determine whether a source is
 * visible from a particular station.
 *
 * @param[in] num_sources The number of source positions.
 * @param[in] d_l         Source l-direction cosines relative to phase centre.
 * @param[in] d_m         Source m-direction cosines relative to phase centre.
 * @param[in] d_n         Source n-direction cosines relative to phase centre.
 * @param[in] l_mul       Factor by which to multiply l-direction cosine.
 * @param[in] m_mul       Factor by which to multiply m-direction cosine.
 * @param[in] n_mul       Factor by which to multiply n-direction cosine.
 * @param[in,out] d_mask  The input and output mask vector.
 */
OSKAR_EXPORT
void oskar_update_horizon_mask_cuda_d(int num_sources, const double* d_l,
        const double* d_m, const double* d_n, const double l_mul,
        const double m_mul, const double n_mul, int* d_mask);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_UPDATE_HORIZON_MASK_CUDA_H_ */
