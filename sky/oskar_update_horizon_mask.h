/*
 * Copyright (c) 2013, The University of Oxford
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

#ifndef OSKAR_UPDATE_HORIZON_MASK_H_
#define OSKAR_UPDATE_HORIZON_MASK_H_

/**
 * @file oskar_update_horizon_mask.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Updates the horizon mask based on the vertical direction cosine
 * (single precision).
 *
 * @details
 * This kernel updates the horizon mask to determine whether a source is
 * visible from a particular station.
 *
 * The operation performed is simply:
 *
 * mask |= (condition > 0)
 *
 * @param[in] num_sources The number of source positions.
 * @param[in,out] mask    The input and output mask vector.
 * @param[in] condition   The vector of conditions to test.
 */
OSKAR_EXPORT
void oskar_update_horizon_mask_f(int num_sources, int* mask,
        const float* condition);

/**
 * @brief
 * Updates the horizon mask based on the vertical direction cosine
 * (double precision).
 *
 * @details
 * This kernel updates the horizon mask to determine whether a source is
 * visible from a particular station.
 *
 * The operation performed is simply:
 *
 * mask |= (condition > 0)
 *
 * @param[in] num_sources The number of source positions.
 * @param[in,out] mask    The input and output mask vector.
 * @param[in] condition   The vector of conditions to test.
 */
OSKAR_EXPORT
void oskar_update_horizon_mask_d(int num_sources, int* mask,
        const double* condition);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_UPDATE_HORIZON_MASK_H_ */
