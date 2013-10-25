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

#ifndef OSKAR_COORDS_CONVERT_GALACTIC_TO_FK5_H_
#define OSKAR_COORDS_CONVERT_GALACTIC_TO_FK5_H_

/**
 * @file oskar_convert_galactic_to_fk5.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert Galactic to FK5 (J2000) equatorial coordinates (single precision).
 *
 * @details
 * This function converts Galactic to FK5 (J2000) equatorial coordinates.
 *
 * @param[in]  num_points The number of points to transform.
 * @param[in]  l          The Galactic longitudes in radians.
 * @param[in]  b          The Galactic latitudes in radians.
 * @param[out] ra         The FK5 (J2000) equatorial Right Ascensions in radians.
 * @param[out] dec        The FK5 (J2000) equatorial Declination in radians.
 */
OSKAR_EXPORT
void oskar_convert_galactic_to_fk5_f(int num_points, const float* l,
        const float* b, float* ra, float* dec);

/**
 * @brief
 * Convert Galactic to FK5 (J2000) equatorial coordinates (double precision).
 *
 * @details
 * This function converts Galactic to FK5 (J2000) equatorial coordinates.
 *
 * @param[in]  num_points The number of points to transform.
 * @param[in]  l          The Galactic longitudes in radians.
 * @param[in]  b          The Galactic latitudes in radians.
 * @param[out] ra         The FK5 (J2000) equatorial Right Ascensions in radians.
 * @param[out] dec        The FK5 (J2000) equatorial Declination in radians.
 */
OSKAR_EXPORT
void oskar_convert_galactic_to_fk5_d(int num_points, const double* l,
        const double* b, double* ra, double* dec);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_GALACTIC_TO_FK5_H_ */
