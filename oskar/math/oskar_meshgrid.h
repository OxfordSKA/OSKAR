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

#ifndef OSKAR_MESHGRID_H_
#define OSKAR_MESHGRID_H_

/**
 * @file oskar_meshgrid.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Generates coordinates for a 2-D grid from a pair of 1-D vectors
 * (single precision).
 *
 * @details
 * This function is equivalent to the 2-D version of the MATLAB meshgrid function.
 *
 * Warning: Arrays \p X and \p Y must be preallocated to length (nx x ny).
 *
 * @param[out] X    2-D grid coordinate array in the x direction.
 * @param[out] Y    2-D grid coordinate array in the y direction.
 * @param[in]  x    1-D vector of x coordinates.
 * @param[in]  nx   Length of the x coordinate array.
 * @param[in]  y    1-D vector of y coordinates.
 * @param[in]  ny   Length of the x coordinate array.
 */
OSKAR_EXPORT
void oskar_meshgrid_d(double* X, double* Y, const double* x, int nx,
        const double* y, int ny);

/**
 * @brief Generates coordinates for a 2-D grid from a pair of 1-D vectors
 * (double precision).
 *
 * @details
 * This function is equivalent to the 2-D version of the MATLAB meshgrid function.
 *
 * Warning: Arrays \p X and \p Y must be preallocated to length (nx x ny).
 *
 * @param[out] X    2-D grid coordinate array in the x direction.
 * @param[out] Y    2-D grid coordinate array in the y direction.
 * @param[in]  x    1-D vector of x coordinates.
 * @param[in]  nx   Length of the x coordinate array.
 * @param[in]  y    1-D vector of y coordinates.
 * @param[in]  ny   Length of the x coordinate array.
 */
OSKAR_EXPORT
void oskar_meshgrid_f(float* X, float* Y, const float* x, int nx,
        const float* y, int ny);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_MESHGRID_H_ */
