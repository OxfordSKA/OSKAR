/*
 * Copyright (c) 2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_CONVERT_APPARENT_RA_DEC_TO_ENU_DIRECTIONS_H_
#define OSKAR_CONVERT_APPARENT_RA_DEC_TO_ENU_DIRECTIONS_H_

/**
 * @file oskar_convert_apparent_ra_dec_to_enu_directions.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert equatorial coordinates to horizontal direction cosines.
 *
 * @details
 * This function computes the direction cosines in the horizontal
 * coordinate system for points specified in an equatorial frame.
 *
 * Points where z is negative are below the local horizon.
 *
 * @param[in]  num_points   The number of points to convert.
 * @param[in]  ra_rad       Right Ascensions in radians.
 * @param[in]  dec_rad      Declinations in radians.
 * @param[in]  lst_rad      The current local sidereal time in radians.
 * @param[in]  latitude_rad The geodetic latitude of the origin.
 * @param[in]  offset_out   Start offset into output arrays.
 * @param[out] x            x-direction-cosines in the horizontal system.
 * @param[out] y            y-direction-cosines in the horizontal system.
 * @param[out] z            z-direction-cosines in the horizontal system.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_convert_apparent_ra_dec_to_enu_directions(
        int num_points, const oskar_Mem* ra_rad, const oskar_Mem* dec_rad,
        double lst_rad, double latitude_rad, int offset_out,
        oskar_Mem* x, oskar_Mem* y, oskar_Mem* z, int* status);

OSKAR_EXPORT
void oskar_convert_apparent_ra_dec_to_enu_directions_float(
        int num_points, const float* ra_rad, const float* dec_rad,
        float lst_rad, float sin_lat, float cos_lat, int offset_out,
        float* x, float* y, float* z);

OSKAR_EXPORT
void oskar_convert_apparent_ra_dec_to_enu_directions_double(
        int num_points, const double* ra_rad, const double* dec_rad,
        double lst_rad, double sin_lat, double cos_lat, int offset_out,
        double* x, double* y, double* z);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
