/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#ifndef OSKAR_ROTATE_H_
#define OSKAR_ROTATE_H_

/**
 * @file oskar_rotate.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif


/**
 * @brief
 * Rotate on a sphere by the specified longitude and latitude
 * (single precision).
 *
 * @details
 * Rotates points on the surface of a sphere by the specified longitude
 * and latitude.
 *
 * @param[in] num_points   Number of points to rotate.
 * @param[in,out] x        The x-coordinates.
 * @param[in,out] y        The y-coordinates.
 * @param[in,out] z        The z-coordinates.
 * @param[in] lon          East-positive longitude rotation angle, in radians.
 * @param[in] lat          North-positive latitude rotation angle, in radians.
 */
OSKAR_EXPORT
void oskar_rotate_sph_f(int num_points, float* x, float* y, float*z,
        float lon, float lat);

/**
 * @brief
 * Rotate on a sphere by the specified longitude and latitude
 * (double precision).
 *
 * @details
 * Rotates points on the surface of a sphere by the specified longitude
 * and latitude.
 *
 * @param[in] num_points   Number of points to rotate.
 * @param[in,out] x        The x-coordinates.
 * @param[in,out] y        The y-coordinates.
 * @param[in,out] z        The z-coordinates.
 * @param[in] lon          East-positive longitude rotation angle, in radians.
 * @param[in] lat          North-positive latitude rotation angle, in radians.
 */
OSKAR_EXPORT
void oskar_rotate_sph_d(int num_points, double* x, double* y, double*z,
        double lon, double lat);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_ROTATE_H_ */
