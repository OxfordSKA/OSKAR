/*
 * Copyright (c) 2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_CONVERT_ANY_TO_ENU_DIRECTIONS_H_
#define OSKAR_CONVERT_ANY_TO_ENU_DIRECTIONS_H_

/**
 * @file oskar_convert_any_to_enu_directions.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert any coordinates to horizontal direction cosines.
 *
 * @details
 * This function converts any supported coordinates to direction cosines
 * in the horizontal coordinate system.
 *
 * @param[in]  coord_type   The enumerated coordinate type (from oskar_global.h).
 * @param[in]  num_points   The number of points to convert.
 * @param[in]  coords       Input coordinates.
 * @param[in]  ref_lon_rad  Reference longitude in radians.
 * @param[in]  ref_lat_rad  Reference latitude in radians.
 * @param[in]  lst_rad      Local Sidereal Time in radians.
 * @param[in]  lat_rad      Geodetic latitude in radians.
 * @param[in]  enu          Output direction cosines.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_convert_any_to_enu_directions(
        int coord_type,
        int num_points,
        const oskar_Mem* const coords[3],
        double ref_lon_rad,
        double ref_lat_rad,
        double lst_rad,
        double lat_rad,
        oskar_Mem* enu[3],
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
