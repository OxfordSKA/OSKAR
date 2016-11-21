/*
 * Copyright (c) 2016, The University of Oxford
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 * 1. Redistributions of source code must retain the above copyright notice,
 *    this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 *    this list of conditions and the following disclaimer in the documentation
 *    and/or other materials provided with the distribution.
 * 3. Neither the name of the University of Oxford nor the names of its
 *    contributors may be used to endorse or promote products derived from this
 *    software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 * LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 * CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 * SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 * INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 * ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 * POSSIBILITY OF SUCH DAMAGE.
 */

#ifndef OSKAR_TELESCOPE_SET_STATION_COORDS_ENU_H_
#define OSKAR_TELESCOPE_SET_STATION_COORDS_ENU_H_

/**
 * @file oskar_telescope_set_station_coords_enu.h
 */

#include <oskar_global.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief
 * Sets station locations with respect to the local tangent plane.
 *
 * @details
 * The coordinate system (ENU, or East-North-Up) is aligned so that the x-axis
 * points to the local geographic East, the y-axis to local geographic North,
 * and the z-axis to the local zenith. The origin is the tangent point to the
 * Earth's ellipsoid.
 *
 * The geodetic longitude and latitude of the origin must also be supplied.
 *
 * @param[in,out] telescope  Telescope model structure to be populated.
 * @param[in] longitude_rad  Telescope centre longitude, in radians.
 * @param[in] latitude_rad   Telescope centre latitude, in radians.
 * @param[in] altitude_m     Telescope centre altitude, in metres.
 * @param[in] num_stations   The number of stations.
 * @param[in] x              Station x coordinates, in metres.
 * @param[in] y              Station y coordinates, in metres.
 * @param[in] z              Station z coordinates, in metres.
 * @param[in] x_err          Station x coordinate error, in metres.
 * @param[in] y_err          Station y coordinate error, in metres.
 * @param[in] z_err          Station z coordinate error, in metres.
 * @param[in,out] status     Status return code.
 */
OSKAR_EXPORT
void oskar_telescope_set_station_coords_enu(oskar_Telescope* telescope,
        double longitude_rad, double latitude_rad, double altitude_m,
        int num_stations, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, const oskar_Mem* x_err, const oskar_Mem* y_err,
        const oskar_Mem* z_err, int* status);

#ifdef __cplusplus
}
#endif

#endif /* OSKAR_TELESCOPE_SET_STATION_COORDS_ENU_H_ */
