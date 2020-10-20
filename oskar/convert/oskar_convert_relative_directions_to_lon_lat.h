/*
 * Copyright (c) 2013-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_CONVERT_RELATIVE_DIRECTIONS_TO_LON_LAT_H_
#define OSKAR_CONVERT_RELATIVE_DIRECTIONS_TO_LON_LAT_H_

/**
 * @file oskar_convert_relative_directions_to_lon_lat.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert direction cosines to angles.
 *
 * @details
 * Returns the longitude and latitude of the supplied array of
 * direction cosines (l, m, n).
 *
 * @param[in]  num_points   Number of positions to evaluate.
 * @param[in]  l            Array of x-positions in cosine space.
 * @param[in]  m            Array of y-positions in cosine space.
 * @param[in]  n            Array of z-positions in cosine space.
 * @param[in]  lon0_rad     Longitude of the field centre, in radians.
 * @param[in]  lat0_rad     Latitude of the field centre, in radians.
 * @param[out] lon_rad      Array of longitude values, in radians.
 * @param[out] lat_rad      Array of latitude values, in radians.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_convert_relative_directions_to_lon_lat(int num_points,
        const oskar_Mem* l, const oskar_Mem* m, const oskar_Mem* n,
        double lon0_rad, double lat0_rad,
        oskar_Mem* lon_rad, oskar_Mem* lat_rad, int* status);

OSKAR_EXPORT
void oskar_convert_relative_directions_to_lon_lat_2d_f(int num_points,
        const float* l, const float* m, const float* n,
        float lon0_rad, float cos_lat0, float sin_lat0,
        float* lon_rad, float* lat_rad);

OSKAR_EXPORT
void oskar_convert_relative_directions_to_lon_lat_2d_d(int num_points,
        const double* l, const double* m, const double* n,
        double lon0_rad, double cos_lat0, double sin_lat0,
        double* lon_rad, double* lat_rad);

OSKAR_EXPORT
void oskar_convert_relative_directions_to_lon_lat_3d_f(int num_points,
        const float* l, const float* m, const float* n,
        float lon0_rad, float cos_lat0, float sin_lat0,
        float* lon_rad, float* lat_rad);

OSKAR_EXPORT
void oskar_convert_relative_directions_to_lon_lat_3d_d(int num_points,
        const double* l, const double* m, const double* n,
        double lon0_rad, double cos_lat0, double sin_lat0,
        double* lon_rad, double* lat_rad);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
