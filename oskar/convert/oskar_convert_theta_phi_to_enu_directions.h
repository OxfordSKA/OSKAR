/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#ifndef OSKAR_CONVERT_THETA_PHI_TO_ENU_DIRECTIONS_H_
#define OSKAR_CONVERT_THETA_PHI_TO_ENU_DIRECTIONS_H_

/**
 * @file oskar_convert_theta_phi_to_enu_directions.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Converts from spherical coordinates to ENU direction cosines.
 *
 * @details
 * Phi is the co-azimuth, measured from x (East) to y (North).
 *  - phi = 0  ==> x,y = 1,0
 *  - phi = 90 ==> x,y = 0,1
 *
 * Theta is the polar (zenith) angle.
 *
 * @param[in]     num    Number of points to convert.
 * @param[in]     theta  Array of theta angles, in radians.
 * @param[in]     phi    Array of phi angles, in radians.
 * @param[out]    x      Array of x (east) direction cosines coordinates.
 * @param[out]    y      Array of y (north) direction cosines coordinates.
 * @param[out]    z      Array of z (up) direction cosines coordinates.
 * @param[in,out] status OSKAR error code.
 */
OSKAR_EXPORT
void oskar_convert_theta_phi_to_enu_directions(unsigned int num,
        const oskar_Mem* theta, const oskar_Mem* phi, oskar_Mem* x,
        oskar_Mem* y, oskar_Mem* z, int* status);

/**
 * @brief Converts from spherical coordinates to ENU direction cosines.
 *
 * @param[in]     num    Number of points to convert.
 * @param[in]     theta  Array of theta angles, in radians.
 * @param[in]     phi    Array of phi angles, in radians.
 * @param[out]    x      Array of x (east) direction cosines coordinates.
 * @param[out]    y      Array of y (north) direction cosines coordinates.
 * @param[out]    z      Array of z (up) direction cosines coordinates.
 */
OSKAR_EXPORT
void oskar_convert_theta_phi_to_enu_directions_d(unsigned int num,
        const double* theta, const double* phi, double* x, double* y,
        double* z);

/**
 * @brief Converts from spherical coordinates to ENU direction cosines.
 *
 * @param[in]     num    Number of points to convert.
 * @param[in]     theta  Array of theta angles, in radians.
 * @param[in]     phi    Array of phi angles, in radians.
 * @param[out]    x      Array of x (east) direction cosines coordinates.
 * @param[out]    y      Array of y (north) direction cosines coordinates.
 * @param[out]    z      Array of z (up) direction cosines coordinates.
 */
OSKAR_EXPORT
void oskar_convert_theta_phi_to_enu_directions_f(unsigned int num,
        const float* theta, const float* phi, float* x, float* y,
        float* z);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_THETA_PHI_TO_ENU_DIRECTIONS_H_ */
