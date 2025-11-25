/*
 * Copyright (c) 2011-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_FILTER_BY_RADIUS_H_
#define OSKAR_SKY_FILTER_BY_RADIUS_H_

/**
 * @file oskar_sky_filter_by_radius.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Removes sources outside given limits.
 *
 * @details
 * This function removes sources from a sky model that lie within
 * the inner radius or beyond the outer radius.
 *
 * @param[in,out] sky          Pointer to sky model.
 * @param[in] inner_radius_rad Inner radius in radians.
 * @param[in] outer_radius_rad Outer radius in radians.
 * @param[in] ra0_rad          Right ascension of the phase centre in radians.
 * @param[in] dec0_rad         Declination of the phase centre in radians.
 * @param[in,out] status       Status return code.
 */
OSKAR_EXPORT
void oskar_sky_filter_by_radius(
        oskar_Sky* sky,
        double inner_radius_rad,
        double outer_radius_rad,
        double ra0_rad,
        double dec0_rad,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
