/*
 * Copyright (c) 2013-2022, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_COORDS_CONVERT_GALACTIC_TO_FK5_H_
#define OSKAR_COORDS_CONVERT_GALACTIC_TO_FK5_H_

/**
 * @file oskar_convert_galactic_to_fk5.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Convert Galactic to FK5 (J2000) equatorial coordinates.
 *
 * @details
 * This function converts Galactic to FK5 (J2000) equatorial coordinates.
 *
 * @param[in]  num_points The number of points to transform.
 * @param[in]  l          The Galactic longitudes in radians.
 * @param[in]  b          The Galactic latitudes in radians.
 * @param[out] ra         The FK5 (J2000) equatorial Right Ascensions in radians.
 * @param[out] dec        The FK5 (J2000) equatorial Declination in radians.
 */
OSKAR_EXPORT
void oskar_convert_galactic_to_fk5(int num_points, const double* l,
        const double* b, double* ra, double* dec);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
