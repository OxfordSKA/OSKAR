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

#ifndef OSKAR_CONVERT_ENU_DIRECTIONS_TO_AZ_EL_H_
#define OSKAR_CONVERT_ENU_DIRECTIONS_TO_AZ_EL_H_

/**
 * @file oskar_convert_enu_directions_to_az_el.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Function to compute azimuth and elevation from horizontal direction
 * cosines (single precision).
 *
 * @details
 * Computes the azimuth and elevation from the given horizontal direction
 * cosines.
 *
 * The directions are:
 * <li> x - pointing East, </li>
 * <li> y - pointing North, </li>
 * <li> z - pointing to the zenith. </li>
 *
 * @param[in]  n  The number of points.
 * @param[in]  x  The x-direction-cosines.
 * @param[in]  y  The y-direction-cosines.
 * @param[in]  z  The z-direction-cosines.
 * @param[out] az The azimuths, in radians.
 * @param[out] el The elevations, in radians.
 */
OSKAR_EXPORT
void oskar_convert_enu_directions_to_az_el_f(int n, const float* x,
        const float* y, const float* z, float* az, float* el);

/**
 * @brief
 * Function to compute azimuth and elevation from horizontal direction
 * cosines (double precision).
 *
 * @details
 * Computes the azimuth and elevation from the given horizontal direction
 * cosines.
 *
 * The directions are:
 * <li> x - pointing East, </li>
 * <li> y - pointing North, </li>
 * <li> z - pointing to the zenith. </li>
 *
 * @param[in]  n  The number of points.
 * @param[in]  x  The x-direction-cosines.
 * @param[in]  y  The y-direction-cosines.
 * @param[in]  z  The z-direction-cosines.
 * @param[out] az The azimuths, in radians.
 * @param[out] el The elevations, in radians.
 */
OSKAR_EXPORT
void oskar_convert_enu_directions_to_az_el_d(int n, const double* x,
        const double* y, const double* z, double* az, double* el);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_ENU_DIRECTIONS_TO_AZ_EL_H_ */
