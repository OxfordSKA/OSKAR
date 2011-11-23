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

#ifndef OSKAR_PHI_THETA_TO_NORMALISED_CUDA_H_
#define OSKAR_PHI_THETA_TO_NORMALISED_CUDA_H_

/**
 * @file oskar_phi_theta_to_normalised_cuda.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Function to convert spherical (phi, theta) to normalised coordinates
 * (single precision).
 *
 * @details
 * This function converts from spherical (phi, theta) to normalised
 * coordinates [0, 1) suitable for a look-up table.
 *
 * All pointers are device pointers.
 *
 * @param[in] n             Number of positions.
 * @param[in] d_phi         Array of input phi coordinates.
 * @param[in] d_theta       Array of input theta coordinates.
 * @param[in] min_phi       Minimum phi coordinate.
 * @param[in] min_theta     Minimum theta coordinate.
 * @param[in] range_phi     Range in phi coordinate.
 * @param[in] range_theta   Range in theta coordinate.
 * @param[out] d_norm_phi   Array of output normalised phi coordinates.
 * @param[out] d_norm_theta Array of output normalised theta coordinates.
 */
OSKAR_EXPORT
int oskar_phi_theta_to_normalised_cuda_f(int n, const float* d_phi,
        const float* d_theta, float min_phi, float min_theta, float range_phi,
        float range_theta, float* d_norm_phi, float* d_norm_theta);

/**
 * @brief
 * Function to convert spherical (phi, theta) to normalised coordinates
 * (double precision).
 *
 * @details
 * This function converts from spherical (phi, theta) to normalised
 * coordinates [0, 1) suitable for a look-up table.
 *
 * All pointers are device pointers.
 *
 * @param[in] n             Number of positions.
 * @param[in] d_phi         Array of input phi coordinates.
 * @param[in] d_theta       Array of input theta coordinates.
 * @param[in] min_phi       Minimum phi coordinate.
 * @param[in] min_theta     Minimum theta coordinate.
 * @param[in] range_phi     Range in phi coordinate.
 * @param[in] range_theta   Range in theta coordinate.
 * @param[out] d_norm_phi   Array of output normalised phi coordinates.
 * @param[out] d_norm_theta Array of output normalised theta coordinates.
 */
OSKAR_EXPORT
int oskar_phi_theta_to_normalised_cuda_d(int n, const double* d_phi,
        const double* d_theta, double min_phi, double min_theta,
        double range_phi, double range_theta, double* d_norm_phi,
        double* d_norm_theta);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_PHI_THETA_TO_NORMALISED_CUDA_H_ */
