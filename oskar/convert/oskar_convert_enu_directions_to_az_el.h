/*
 * Copyright (c) 2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_CONVERT_ENU_DIRECTIONS_TO_AZ_EL_H_
#define OSKAR_CONVERT_ENU_DIRECTIONS_TO_AZ_EL_H_

/**
 * @file oskar_convert_enu_directions_to_az_el.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert horizontal direction cosines to (azimuth, elevation).
 *
 * @details
 * This function converts the direction cosines in the horizontal
 * coordinate system to (azimuth, elevation) angles.
 *
 * @param[in]  num_points   The number of points to convert.
 * @param[in]  x            x-direction-cosines in the horizontal system.
 * @param[in]  y            y-direction-cosines in the horizontal system.
 * @param[in]  z            z-direction-cosines in the horizontal system.
 * @param[out] az_rad       Azimuth values in radians.
 * @param[out] el_rad       Elevation values in radians.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_convert_enu_directions_to_az_el(
        int num_points, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, oskar_Mem* az_rad, oskar_Mem* el_rad, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
