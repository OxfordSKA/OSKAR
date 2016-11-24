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

#include "math/oskar_cmath.h"
#include "convert/oskar_convert_ecef_to_enu.h"
#include "convert/oskar_convert_geodetic_spherical_to_ecef.h"
#include "telescope/oskar_telescope.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_set_station_coords_wgs84(oskar_Telescope* telescope,
        double longitude_rad, double latitude_rad, double altitude_m,
        int num_stations, const oskar_Mem* lon_deg, const oskar_Mem* lat_deg,
        const oskar_Mem* alt_m, int* status)
{
    int i;

    /* Check lengths. */
    if ((int)oskar_mem_length(lon_deg) < num_stations ||
            (int)oskar_mem_length(lat_deg) < num_stations ||
            (int)oskar_mem_length(alt_m) < num_stations)
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
        /* longitude, latitude, altitude */
        double wgs[] = {0.0, 0.0, 0.0};
        double ecef[] = {0.0, 0.0, 0.0};
        double hor[] = {0.0, 0.0, 0.0};

        /* Get coordinates from input arrays. */
        wgs[0] = oskar_mem_get_element(lon_deg, i, status);
        wgs[1] = oskar_mem_get_element(lat_deg, i, status);
        wgs[2] = oskar_mem_get_element(alt_m, i, status);

        /* Convert geodetic spherical to ECEF. */
        wgs[0] *= M_PI / 180.0;
        wgs[1] *= M_PI / 180.0;
        oskar_station_set_position(oskar_telescope_station(telescope, i),
                wgs[0], wgs[1], wgs[2]);
        oskar_convert_geodetic_spherical_to_ecef(1, &wgs[0], &wgs[1], &wgs[2],
                &ecef[0], &ecef[1], &ecef[2]);

        /* Convert station ECEF to horizon plane coordinates. */
        oskar_convert_ecef_to_enu(1, &ecef[0], &ecef[1], &ecef[2],
                longitude_rad, latitude_rad, altitude_m, &hor[0], &hor[1], &hor[2]);

        /* Store the offset geocentric and horizon plane coordinates. */
        oskar_telescope_set_station_coords(telescope, i, &ecef[0], &ecef[0],
                &hor[0], &hor[0], status);
        if (*status) break;
    }

    /* (Re-)Set unique station IDs. */
    oskar_telescope_set_station_ids(telescope);
}

#ifdef __cplusplus
}
#endif
