/*
 * Copyright (c) 2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_CONVERT_ENU_DIRECTIONS_TO_LOCAL_TANGENT_PLANE_H_
#define OSKAR_CONVERT_ENU_DIRECTIONS_TO_LOCAL_TANGENT_PLANE_H_

/**
 * @file oskar_convert_enu_directions_to_local_tangent_plane.h
 */

#include <oskar_global.h>
#include <mem/oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert horizontal direction cosines to local tangent plane coordinates.
 *
 * @details
 * This function converts the direction cosines in the horizontal
 * coordinate system to 2D direction cosines on a local tangent plane,
 * specified by the reference coordinates.
 *
 * @param[in]  num_points   The number of points to convert.
 * @param[in]  x            x-direction-cosines in the horizontal system.
 * @param[in]  y            y-direction-cosines in the horizontal system.
 * @param[in]  z            z-direction-cosines in the horizontal system.
 * @param[in]  ref_az_rad   Reference azimuth in radians.
 * @param[in]  ref_el_rad   Reference elevation in radians.
 * @param[out] l            l-direction-cosines on the local tangent plane.
 * @param[out] m            m-direction-cosines on the local tangent plane.
 * @param[in,out] status    Status return code.
 */
OSKAR_EXPORT
void oskar_convert_enu_directions_to_local_tangent_plane(int num_points,
        const oskar_Mem* x, const oskar_Mem* y, const oskar_Mem* z,
        double ref_az_rad, double ref_el_rad, oskar_Mem* l, oskar_Mem* m,
        int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
