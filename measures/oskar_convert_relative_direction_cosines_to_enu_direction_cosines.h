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

#ifndef OSKAR_CONVERT_RELATIVE_DIRECTION_COSINES_TO_ENU_DIRECTION_COSINES_H_
#define OSKAR_CONVERT_RELATIVE_DIRECTION_COSINES_TO_ENU_DIRECTION_COSINES_H_

/**
 * @file oskar_convert_relative_direction_cosines_to_enu_direction_cosines.h
 */

#include <oskar_global.h>
#include <oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Converts from relative direction cosines to horizon ENU direction cosines.
 *
 * @details
 *
 * @param[out]    y      ENU direction cosines.
 * @param[out]    z      ENU direction cosines.
 * @param[in]     np     Number of points to convert.
 * @param[in]     l      Relative direction cosines.
 * @param[in]     m      Relative direction cosines.
 * @param[in]     n      Relative direction cosines.
 * @param[in]     ra0    Right Ascension of the origin of the relative.
 *                       directions, in radians.
 * @param[in]     dec0   Declination of the origin of the relative directions,
 *                       in radians.
 * @param[in]     LAST   Local apparent sideral time.
 * @param[in]     lat    Latitude of the ENU coordinate frame, in radians.
 * @param[in/out] status Error status code.
 */
OSKAR_EXPORT
void oskar_convert_relative_direction_cosines_to_enu_direction_cosines(
        oskar_Mem* x, oskar_Mem* y, oskar_Mem* z, int np, const oskar_Mem* l,
        const oskar_Mem* m, const oskar_Mem* n, const double ra0, double dec0,
        double LAST, double lat, int* status);

/**
 * @brief
 * Converts from relative direction cosines to horizon ENU direction cosines.
 * (double precision, CPU (openMP) version)
 *
 * @details
 *
 * @param[out]    y      ENU direction cosines.
 * @param[out]    z      ENU direction cosines.
 * @param[in]     np     Number of points to convert.
 * @param[in]     l      Relative direction cosines.
 * @param[in]     m      Relative direction cosines.
 * @param[in]     n      Relative direction cosines.
 * @param[in]     ra0    Right Ascension of the origin of the relative.
 *                       directions, in radians.
 * @param[in]     dec0   Declination of the origin of the relative directions,
 *                       in radians.
 * @param[in]     LAST   Local apparent sideral time.
 * @param[in]     lat    Latitude of the ENU coordinate frame, in radians.
 * @param[in/out] status Error status code.
 */
OSKAR_EXPORT
void oskar_convert_relative_direction_cosines_to_enu_direction_cosines_d(
        double* x, double* y, double* z, int np, const double* l,
        const double* m, const double* n, double ra0, double dec0, double LAST,
        double lat);

/**
 * @brief
 * Converts from relative direction cosines to horizon ENU direction cosines.
 * (single precision, CPU (openMP) version)
 *
 * @details
 *
 * @param[out]    y      ENU direction cosines.
 * @param[out]    z      ENU direction cosines.
 * @param[in]     np     Number of points to convert.
 * @param[in]     l      Relative direction cosines.
 * @param[in]     m      Relative direction cosines.
 * @param[in]     n      Relative direction cosines.
 * @param[in]     ra0    Right Ascension of the origin of the relative.
 *                       directions, in radians.
 * @param[in]     dec0   Declination of the origin of the relative directions,
 *                       in radians.
 * @param[in]     LAST   Local apparent sideral time.
 * @param[in]     lat    Latitude of the ENU coordinate frame, in radians.
 * @param[in/out] status Error status code.
 */
OSKAR_EXPORT
void oskar_convert_relative_direction_cosines_to_enu_direction_cosines_f(
        float* x, float* y, float* z, int np, const float* l, const float* m,
        const float* n, float ra0, float dec0, float LAST, float lat);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_CONVERT_RELATIVE_DIRECTION_COSINES_TO_ENU_DIRECTION_COSINES_H_ */
