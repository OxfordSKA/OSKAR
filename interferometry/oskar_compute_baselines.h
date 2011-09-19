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

#ifndef OSKAR_INTERFEROMETRY_COMPUTE_BASELINES_H_
#define OSKAR_INTERFEROMETRY_COMPUTE_BASELINES_H_

/**
 * @file oskar_compute_baselines.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Computes the baseline coordinates for all station pairs
 * (single precision).
 *
 * @details
 * Given the (u,v,w) coordinates for each station, this function computes
 * the baseline coordinates for all station pairs.
 *
 * The output arrays must be pre-sized to length na * (na - 1) / 2.
 *
 * @param[in]  na The number of stations.
 * @param[in]  au The station u-positions (length na).
 * @param[in]  av The station v-positions (length na).
 * @param[in]  aw The station w-positions (length na).
 * @param[out] bu The baseline u-positions (length na * (na - 1) / 2).
 * @param[out] bv The baseline v-positions (length na * (na - 1) / 2).
 * @param[out] bw The baseline w-positions (length na * (na - 1) / 2).
 */
OSKAR_EXPORT
void oskar_compute_baselines_f(int na, const float* au,
        const float* av, const float* aw, float* bu, float* bv, float* bw);

/**
 * @brief
 * Computes the baseline coordinates for all station pairs
 * (double precision).
 *
 * @details
 * Given the (u,v,w) coordinates for each station, this function computes
 * the baseline coordinates for all station pairs.
 *
 * The output arrays must be pre-sized to length na * (na - 1) / 2.
 *
 * @param[in]  na The number of stations.
 * @param[in]  au The station u-positions (length na).
 * @param[in]  av The station v-positions (length na).
 * @param[in]  aw The station w-positions (length na).
 * @param[out] bu The baseline u-positions (length na * (na - 1) / 2).
 * @param[out] bv The baseline v-positions (length na * (na - 1) / 2).
 * @param[out] bw The baseline w-positions (length na * (na - 1) / 2).
 */
OSKAR_EXPORT
void oskar_compute_baselines_d(int na, const double* au,
        const double* av, const double* aw, double* bu, double* bv, double* bw);

#ifdef __cplusplus
}
#endif

#endif // OSKAR_INTERFEROMETRY_COMPUTE_BASELINES_H_
