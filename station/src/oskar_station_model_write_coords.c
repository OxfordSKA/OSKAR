/*
 * Copyright (c) 2011, The University of Oxford
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

#include "station/oskar_station_model_location.h"
#include "station/oskar_station_model_type.h"
#include "station/oskar_station_model_write_coords.h"
#include "stdio.h"
#include "stdlib.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_station_model_write_coords(const char* filename,
        const oskar_StationModel* station)
{
    int i, location, type;
    FILE* file;

    if (filename == NULL || station == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Get type and location. */
    type = oskar_station_model_type(station);
    location = oskar_station_model_location(station);
    if (location != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    if (station->coord_units != OSKAR_METRES)
        return OSKAR_ERR_BAD_UNITS;

    file = fopen(filename, "w");
    if (!file)
        return OSKAR_ERR_FILE_IO;

    fprintf(file, "# Number of antennas  = %i\n", station->num_elements);
    fprintf(file, "# Longitude [radians] = %f\n", station->longitude_rad);
    fprintf(file, "# Latitude [radians]  = %f\n", station->latitude_rad);
    fprintf(file, "# Altitude [metres]   = %f\n", station->altitude_metres);
    fprintf(file, "# Local horizontal x(east), y(north), z(zenith), delta_x, delta_y, delta_z [metres]\n");
    if (type == OSKAR_DOUBLE)
    {
        double x, y, z, delta_x, delta_y, delta_z;
        for (i = 0; i < station->num_elements; ++i)
        {
            x = ((double*)station->x_weights.data)[i];
            y = ((double*)station->y_weights.data)[i];
            z = ((double*)station->z_weights.data)[i];
            delta_x = ((double*)station->x_signal.data)[i] - x;
            delta_y = ((double*)station->y_signal.data)[i] - y;
            delta_z = ((double*)station->z_signal.data)[i] - z;
            fprintf(file, "% -12.6e,% -12.6e,% -12.6e,% -12.6e,% -12.6e,% -12.6e\n",
                    x, y, z, delta_x, delta_y, delta_z);
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        float x, y, z, delta_x, delta_y, delta_z;
        for (i = 0; i < station->num_elements; ++i)
        {
            x = ((float*)station->x_weights.data)[i];
            y = ((float*)station->y_weights.data)[i];
            z = ((float*)station->z_weights.data)[i];
            delta_x = ((float*)station->x_signal.data)[i] - x;
            delta_y = ((float*)station->y_signal.data)[i] - y;
            delta_z = ((float*)station->z_signal.data)[i] - z;
            fprintf(file, "% -12.6e,% -12.6e,% -12.6e,% -12.6e,% -12.6e,% -12.6e\n",
                    x, y, z, delta_x, delta_y, delta_z);
        }
    }
    else
    {
        return OSKAR_ERR_BAD_DATA_TYPE;
    }

    fclose(file);
    return 0;
}

#ifdef __cplusplus
}
#endif
