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

#include "convert/oskar_convert_ecef_to_geodetic_spherical.h"
#include "convert/oskar_convert_enu_to_offset_ecef.h"
#include "convert/oskar_convert_offset_ecef_to_ecef.h"
#include "telescope/oskar_telescope.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_set_station_coords_enu(oskar_Telescope* telescope,
        double longitude_rad, double latitude_rad, double altitude_m,
        int num_stations, const oskar_Mem* x, const oskar_Mem* y,
        const oskar_Mem* z, const oskar_Mem* x_err, const oskar_Mem* y_err,
        const oskar_Mem* z_err, int* status)
{
    int i;

    /* Check lengths. */
    if ((int)oskar_mem_length(x) < num_stations ||
            (int)oskar_mem_length(y) < num_stations ||
            (int)oskar_mem_length(z) < num_stations ||
            (int)oskar_mem_length(x_err) < num_stations ||
            (int)oskar_mem_length(y_err) < num_stations ||
            (int)oskar_mem_length(z_err) < num_stations)
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Resize the telescope model to hold the station data. */
    oskar_telescope_resize(telescope, num_stations, status);
    if (*status) return;

    /* Store the telescope centre longitude, latitude, and altitude. */
    oskar_telescope_set_position(telescope,
            longitude_rad, latitude_rad, altitude_m);

    /* Loop over station coordinates. */
    for (i = 0; i < num_stations; ++i)
    {
        double lon = 0.0, lat = 0.0, alt = 0.0;

        /* x, y, z, delta x, delta y, delta z */
        double hor[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double ecef[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        /* Get coordinates from input arrays. */
        hor[0] = oskar_mem_get_element(x, i, status);
        hor[1] = oskar_mem_get_element(y, i, status);
        hor[2] = oskar_mem_get_element(z, i, status);
        hor[3] = oskar_mem_get_element(x_err, i, status);
        hor[4] = oskar_mem_get_element(y_err, i, status);
        hor[5] = oskar_mem_get_element(z_err, i, status);

        /* Get "true" coordinates ([3, 4, 5]) from "measured" coordinates. */
        hor[3] += hor[0];
        hor[4] += hor[1];
        hor[5] += hor[2];

        /* Convert horizon plane to offset geocentric Cartesian coordinates. */
        oskar_convert_enu_to_offset_ecef_d(1, &hor[0], &hor[1], &hor[2],
                longitude_rad, latitude_rad, &ecef[0], &ecef[1], &ecef[2]);
        oskar_convert_enu_to_offset_ecef_d(1, &hor[3], &hor[4], &hor[5],
                longitude_rad, latitude_rad, &ecef[3], &ecef[4], &ecef[5]);

        /* Store the offset geocentric and horizon plane coordinates. */
        oskar_telescope_set_station_coords(telescope, i, &ecef[0], &ecef[3],
                &hor[0], &hor[3], status);
        if (*status) break;

        /* Convert to ECEF, then to station longitude, latitude, altitude. */
        oskar_convert_offset_ecef_to_ecef(1, &ecef[3], &ecef[4], &ecef[5],
                longitude_rad, latitude_rad, altitude_m,
                &ecef[3], &ecef[4], &ecef[5]);
        oskar_convert_ecef_to_geodetic_spherical(1, &ecef[3], &ecef[4],
                &ecef[5], &lon, &lat, &alt);
        oskar_station_set_position(oskar_telescope_station(telescope, i),
                lon, lat, alt);
    }

    /* (Re-)Set unique station IDs. */
    oskar_telescope_set_station_ids(telescope);
}

#ifdef __cplusplus
}
#endif
