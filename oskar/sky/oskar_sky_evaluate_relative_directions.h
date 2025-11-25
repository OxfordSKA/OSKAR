/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_EVALUATE_RELATIVE_DIRECTIONS_H_
#define OSKAR_SKY_EVALUATE_RELATIVE_DIRECTIONS_H_

/**
 * @file oskar_sky_evaluate_relative_directions.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluates 3D direction cosines of sources relative to phase centre.
 *
 * @details
 * This function populates the 3D direction cosines (l,m,n coordinates)
 * of all sources relative to the phase centre.
 *
 * It assumes that the source RA and Dec positions have already been filled.
 *
 * @param[in,out] sky    Pointer to sky model structure.
 * @param[in] ra0_rad    Right Ascension of phase centre, in radians.
 * @param[in] dec0_rad   Declination of phase centre, in radians.
 * @param[in,out] status Status return code.
 */
OSKAR_EXPORT
void oskar_sky_evaluate_relative_directions(
        oskar_Sky* sky,
        double ra0_rad,
        double dec0_rad,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
