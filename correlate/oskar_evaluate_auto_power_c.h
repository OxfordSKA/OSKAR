/*
 * Copyright (c) 2015, The University of Oxford
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

#ifndef OSKAR_EVALUATE_AUTO_POWER_C_H_
#define OSKAR_EVALUATE_AUTO_POWER_C_H_

/**
 * @file oskar_evaluate_auto_power_c.h
 */

#include <oskar_global.h>
#include <oskar_vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Function to evaluate the auto-power product from all stations
 * (single precision).
 *
 * @details
 * This function evaluates the average auto-power product for the supplied
 * sources from all stations.
 *
 * @param[in] num_sources    The number of sources in the input arrays.
 * @param[in] jones          Pointer to Jones matrix list
 *                           (length \p num_sources).
 * @param[out] out           Pointer to output auto-power product
 *                           (length \p num_sources).
 */
OSKAR_EXPORT
void oskar_evaluate_auto_power_f(const int num_sources,
        const float4c* restrict jones, float4c* restrict out);

/**
 * @brief
 * Function to evaluate the auto-power product from all stations
 * (single precision).
 *
 * @details
 * This function evaluates the average auto-power product for the supplied
 * sources from all stations.
 *
 * @param[in] num_sources    The number of sources in the input arrays.
 * @param[in] jones          Pointer to Jones matrix list
 *                           (length \p num_sources).
 * @param[out] out           Pointer to output auto-power product
 *                           (length \p num_sources).
 */
OSKAR_EXPORT
void oskar_evaluate_auto_power_scalar_f(const int num_sources,
        const float2* restrict jones, float2* restrict out);

/**
 * @brief
 * Function to evaluate the auto-power product from all stations
 * (double precision).
 *
 * @details
 * This function evaluates the average auto-power product for the supplied
 * sources from all stations.
 *
 * @param[in] num_sources    The number of sources in the input arrays.
 * @param[in] jones          Pointer to Jones matrix list
 *                           (length \p num_sources).
 * @param[out] out           Pointer to output auto-power product
 *                           (length \p num_sources).
 */
OSKAR_EXPORT
void oskar_evaluate_auto_power_d(const int num_sources,
        const double4c* restrict jones, double4c* restrict out);

/**
 * @brief
 * Function to evaluate the auto-power product from all stations
 * (double precision).
 *
 * @details
 * This function evaluates the average auto-power product for the supplied
 * sources from all stations.
 *
 * @param[in] num_sources    The number of sources in the input arrays.
 * @param[in] jones          Pointer to Jones matrix list
 *                           (length \p num_sources).
 * @param[out] out           Pointer to output auto-power product
 *                           (length \p num_sources).
 */
OSKAR_EXPORT
void oskar_evaluate_auto_power_scalar_d(const int num_sources,
        const double2* restrict jones, double2* restrict out);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_AUTO_POWER_C_H_ */
