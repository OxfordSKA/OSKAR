/*
 * Copyright (c) 2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_CONVERT_AZ_EL_TO_ENU_DIRECTIONS_H_
#define OSKAR_CONVERT_AZ_EL_TO_ENU_DIRECTIONS_H_

/**
 * @file oskar_convert_az_el_to_enu_directions.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert (azimuth, elevation) to horizontal direction cosines.
 *
 * @details
 * This function converts the (azimuth, elevation) angles to
 * direction cosines in the horizontal coordinate system.
 *
 * @param[in]  num_points   The number of points to convert.
 * @param[in]  az_rad       Azimuth values in radians.
 * @param[in]  el_rad       Elevation values in radians.
 * @param[out] x            x-direction-cosines in the horizontal system.
 * @param[out] y            y-direction-cosines in the horizontal system.
 * @param[out] z            z-direction-cosines in the horizontal system.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_convert_az_el_to_enu_directions(
        int num_points, const oskar_Mem* az_rad, const oskar_Mem* el_rad,
        oskar_Mem* x, oskar_Mem* y, oskar_Mem* z, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
