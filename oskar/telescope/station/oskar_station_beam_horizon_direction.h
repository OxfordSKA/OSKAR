/*
 * Copyright (c) 2011-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_STATION_BEAM_HORIZON_DIRECTION_H_
#define OSKAR_STATION_BEAM_HORIZON_DIRECTION_H_

/**
 * @file oskar_station_beam_horizon_direction.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Evaluate the beam phase centre coordinates as horizontal direction
 * cosines for the specified Greenwich apparent sidereal time.
 *
 * @details
 * Converts the station beam phase centre in the station model to
 * horizontal (x,y,z) coordinates for the specified time.
 *
 * @param[in]  station   Station model.
 * @param[in]  gast_rad  The Greenwich apparent sidereal time, in radians.
 * @param[out] x         Horizontal x-direction-cosine of the beam phase centre.
 * @param[out] y         Horizontal y-direction-cosine of the beam phase centre.
 * @param[out] z         Horizontal z-direction-cosine of the beam phase centre.
 * @param[in,out] status Status return code.
 */
OSKAR_EXPORT
void oskar_station_beam_horizon_direction(const oskar_Station* station,
        const double gast_rad, double* x, double* y, double* z, int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
