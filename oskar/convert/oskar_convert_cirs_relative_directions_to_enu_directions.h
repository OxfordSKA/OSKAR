/*
 * Copyright (c) 2014-2019, The University of Oxford
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

#ifndef OSKAR_CONVERT_CIRS_RELATIVE_DIRECTIONS_TO_ENU_DIRECTIONS_H_
#define OSKAR_CONVERT_CIRS_RELATIVE_DIRECTIONS_TO_ENU_DIRECTIONS_H_

/**
 * @file oskar_convert_cirs_relative_directions_to_enu_directions.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Converts CIRS relative direction cosines to ENU direction cosines.
 *
 * @details
 * This function converts CIRS vector components relative to
 * the given right ascension and declination to topocentric east-north-up
 * vector components for an observer at the given location.
 *
 * Allowance is made for polar motion and diurnal aberration.
 * The magnitude of the diurnal aberration vector can be obtained by calling
 * oskar_evaluate_diurnal_aberration().
 * If not known, these quantities can be set to zero.
 *
 * No allowance is made for atmospheric refraction.
 *
 * To be rigorous, lon_rad is east longitude + s_prime, the direction to the
 * terrestial intermediate origin (TIO). This correction is sufficiently small
 * to be safely neglected for all but the most precise applications.
 *
 * This function uses the same algorithm as implemented in the IAU
 * Standards of Fundamental Astronomy (SOFA) library.
 *
 * @param[in] num_points         Number of points in all arrays.
 * @param[in] l                  CIRS relative directions (l-component).
 * @param[in] m                  CIRS relative directions (m-component).
 * @param[in] n                  CIRS relative directions (n-component).
 * @param[in] ra0_rad            CIRS reference right ascension, in radians.
 * @param[in] dec0_rad           CIRS reference declination, in radians.
 * @param[in] lon_rad            Observer's longitude, in radians.
 * @param[in] lat_rad            Observer's geodetic latitude, in radians.
 * @param[in] era_rad            Current Earth rotation angle, in radians.
 * @param[in] pm_x_rad           Polar motion, x component, in radians.
 * @param[in] pm_y_rad           Polar motion, y component, in radians.
 * @param[in] diurnal_aberration Magnitude of diurnal aberration vector.
 * @param[out] x                 Output ENU x (east) vector components.
 * @param[out] y                 Output ENU y (north) vector components.
 * @param[out] z                 Output ENU z (up) vector components.
 */
OSKAR_EXPORT
void oskar_convert_cirs_relative_directions_to_enu_directions(
        int at_origin,
        int bypass,
        int offset_in,
        int num_points,
        const oskar_Mem* l,
        const oskar_Mem* m,
        const oskar_Mem* n,
        double ra0_rad,
        double dec0_rad,
        double lon_rad,
        double lat_rad,
        double era_rad,
        double pm_x_rad,
        double pm_y_rad,
        double diurnal_aberration,
        int offset_out,
        oskar_Mem* x,
        oskar_Mem* y,
        oskar_Mem* z,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_CIRS_RELATIVE_DIRECTIONS_TO_ENU_DIRECTIONS_H_ */
