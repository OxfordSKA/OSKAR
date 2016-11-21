/*
 * Copyright (c) 2014, The University of Oxford
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

#ifndef OSKAR_EVALUATE_DIURNAL_ABERRATION_H_
#define OSKAR_EVALUATE_DIURNAL_ABERRATION_H_

/**
 * @file oskar_evaluate_diurnal_aberration.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates the magnitude of the diurnal aberration vector.
 *
 * @details
 * Returns the magnitude of the diurnal aberration vector for an observer
 * at the given location and time.
 *
 * The quantity s_prime (the TIO locator) is sufficiently small that it can
 * be ignored for most applications.
 *
 * This function uses the same algorithm as implemented in the IAU
 * Standards of Fundamental Astronomy (SOFA) library.
 *
 * @param[in] lon_rad   Observer's longitude, in radians.
 * @param[in] lat_rad   Observer's geodetic latitude, in radians.
 * @param[in] height_m  Observer's height above ellipsoid, in metres.
 * @param[in] era_rad   Current Earth rotation angle, in radians.
 * @param[in] pm_x_rad  Polar motion, x component, in radians.
 * @param[in] pm_y_rad  Polar motion, y component, in radians.
 * @param[in] s_prime   The TIO locator, in radians (can be zero).
 */
OSKAR_EXPORT
double oskar_evaluate_diurnal_aberration(double lon_rad, double lat_rad,
        double height_m, double era_rad, double pm_x_rad, double pm_y_rad,
        double s_prime);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_EVALUATE_DIURNAL_ABERRATION_H_ */
