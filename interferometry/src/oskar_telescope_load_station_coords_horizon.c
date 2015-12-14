/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include <private_telescope.h>

#include <oskar_convert_ecef_to_geodetic_spherical.h>
#include <oskar_convert_enu_to_offset_ecef.h>
#include <oskar_convert_offset_ecef_to_ecef.h>
#include <oskar_telescope.h>
#include <oskar_getline.h>
#include <oskar_string_to_array.h>

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_load_station_coords_horizon(oskar_Telescope* telescope,
        const char* filename, double longitude, double latitude,
        double altitude, int* status)
{
    /* Declare the line buffer and counter. */
    char* line = NULL;
    size_t bufsize = 0;
    int n = 0, type = 0;
    FILE* file;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check type. */
    type = oskar_telescope_precision(telescope);
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }

    /* Open the file. */
    file = fopen(filename, "r");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Store the telescope centre longitude, latitude, and altitude. */
    telescope->lon_rad = longitude;
    telescope->lat_rad = latitude;
    telescope->alt_metres = altitude;

    /* Loop over each line in the file. */
    while (oskar_getline(&line, &bufsize, file) != OSKAR_ERR_EOF)
    {
        double lon = 0.0, lat = 0.0, alt = 0.0;

        /* x, y, z, delta x, delta y, delta z */
        double hor[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
        double ecef[] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

        /* Load coordinates. */
        if (oskar_string_to_array_d(line, 6, hor) < 2) continue;

        /* Resize the telescope model to hold the station data.
         * We can't resize to more than needed, since we would then lose track
         * of the actual allocated size of the model when
         * telescope->num_stations = n is finally set. */
        if (telescope->num_stations <= n)
        {
            oskar_telescope_resize(telescope, n + 1, status);
            if (*status) break;
        }

        /* Get "true" coordinates ([3, 4, 5]) from "measured" coordinates. */
        hor[3] += hor[0];
        hor[4] += hor[1];
        hor[5] += hor[2];

        /* Convert horizon plane to offset geocentric Cartesian coordinates. */
        oskar_convert_enu_to_offset_ecef_d(1, &hor[0], &hor[1], &hor[2],
                longitude, latitude, &ecef[0], &ecef[1], &ecef[2]);
        oskar_convert_enu_to_offset_ecef_d(1, &hor[3], &hor[4], &hor[5],
                longitude, latitude, &ecef[3], &ecef[4], &ecef[5]);

        /* Store the offset geocentric and horizon plane coordinates. */
        oskar_telescope_set_station_coords(telescope, n, &ecef[0], &ecef[3],
                &hor[0], &hor[3], status);
        if (*status) break;

        /* Convert to ECEF, then to station longitude, latitude, altitude. */
        oskar_convert_offset_ecef_to_ecef(1, &ecef[3], &ecef[4], &ecef[5],
                longitude, latitude, altitude, &ecef[3], &ecef[4], &ecef[5]);
        oskar_convert_ecef_to_geodetic_spherical(1, &ecef[3], &ecef[4],
                &ecef[5], &lon, &lat, &alt);
        oskar_station_set_position(oskar_telescope_station(telescope, n),
                lon, lat, alt);

        /* Increment counter. */
        ++n;
    }

    /* Record the number of station positions loaded. */
    telescope->num_stations = n;

    /* Free the line buffer and close the file. */
    if (line) free(line);
    fclose(file);
}

#ifdef __cplusplus
}
#endif
