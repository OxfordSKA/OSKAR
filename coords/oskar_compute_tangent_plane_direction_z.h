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

#ifndef OSKAR_COMPUTE_TANGENT_PLANE_DIRECTION_Z_H_
#define OSKAR_COMPUTE_TANGENT_PLANE_DIRECTION_Z_H_

/**
 * @file oskar_compute_tangent_plane_direction_z.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Compute z-direction cosines from x and y (single precision).
 *
 * @details
 * This function computes z-direction cosines from x,y-direction cosines,
 * using the relation z = sqrt(1 - x*x - y*y) - 1.
 *
 * @param[in]  n The number of points.
 * @param[out] x The x-direction-cosines.
 * @param[out] y The y-direction-cosines.
 * @param[out] z The z-direction-cosines.
 */
OSKAR_EXPORT
void oskar_compute_tangent_plane_direction_z_f(int n, const float* x,
        const float* y, float* z);

/**
 * @brief
 * Compute z-direction cosines from x and y (double precision).
 *
 * @details
 * This function computes z-direction cosines from x,y-direction cosines,
 * using the relation z = sqrt(1 - x*x - y*y) - 1.
 *
 * @param[in]  n The number of points.
 * @param[out] x The x-direction-cosines.
 * @param[out] y The y-direction-cosines.
 * @param[out] z The z-direction-cosines.
 */
OSKAR_EXPORT
void oskar_compute_tangent_plane_direction_z_d(int n, const double* x,
        const double* y, double* z);


#ifdef __cplusplus
}
#endif

#endif /* OSKAR_COMPUTE_TANGENT_PLANE_DIRECTION_Z_H_ */
