/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#ifndef OSKAR_TELESCOPE_SET_STATION_COORDS_H_
#define OSKAR_TELESCOPE_SET_STATION_COORDS_H_

/**
 * @file oskar_telescope_set_station_coords.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Sets the coordinates of a station in the telescope model.
 *
 * @details
 * This function sets the coordinates of the specified station in the telescope
 * model. Both the offset ECEF (Earth-centred-Earth-fixed) and horizon
 * frame coordinates must be supplied.
 *
 * All coordinates are in metres.
 *
 * @param[in] dst                     Telescope model structure to update.
 * @param[in] index                   Station index to set.
 * @param[in] true_geodetic[3]        True geodetic station coordinates, in radians/metres (long, lat, altitude).
 * @param[in] measured_offset_ecef[3] Measured (x,y,z) of station (offset ECEF).
 * @param[in] true_offset_ecef[3]     True (x,y,z) of station (offset ECEF).
 * @param[in] measured_enu[3]         Measured (x,y,z) of station (horizon).
 * @param[in] true_enu[3]             True (x,y,z) of station (horizon).
 * @param[in,out] status              Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_set_station_coords(oskar_Telescope* dst, int index,
        const double true_geodetic[3],
        const double measured_offset_ecef[3], const double true_offset_ecef[3],
        const double measured_enu[3], const double true_enu[3], int* status);

#ifdef __cplusplus
}
#endif

#endif /* include guard */
