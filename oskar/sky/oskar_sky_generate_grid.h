/*
 * Copyright (c) 2016-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_SKY_GENERATE_GRID_H_
#define OSKAR_SKY_GENERATE_GRID_H_

/**
 * @file oskar_sky_generate_grid.h
 */

#include "oskar_global.h"

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Generates a grid of sources at the specified tangent point on the sky.
 *
 * @details
 * Generates a grid of sources at the specified tangent point on the sky.
 *
 * @param[in] precision     Precision of the sky model to create.
 * @param[in] ra0_rad       Right Ascension of grid centre, in radians.
 * @param[in] dec0_rad      Declination of grid centre, in radians.
 * @param[in] side_length   Side length of generated grid.
 * @param[in] fov_rad       Grid field-of-view, in radians.
 * @param[in] mean_flux_jy  Mean Stokes-I source flux, in Jy.
 * @param[in] std_flux_jy   Standard deviation Stokes-I source flux, in Jy.
 * @param[in] seed          Random generator seed.
 * @param[in,out]  status   Status return code.
 */
OSKAR_EXPORT
oskar_Sky* oskar_sky_generate_grid(
        int precision,
        double ra0_rad,
        double dec0_rad,
        int side_length,
        double fov_rad,
        double mean_flux_jy,
        double std_flux_jy,
        int seed,
        int* status
);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
