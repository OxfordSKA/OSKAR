/*
 * Copyright (c) 2012, The University of Oxford
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

#include "interferometry/oskar_telescope_model_save_station_coords.h"
#include "interferometry/oskar_telescope_model_location.h"
#include "interferometry/oskar_telescope_model_type.h"
#include "interferometry/oskar_TelescopeModel.h"
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_telescope_model_save_station_coords(
        const oskar_TelescopeModel* telescope, const char* filename)
{
    int i, type, location;
    FILE* file;

    /* Sanity check on inputs. */
    if (telescope == NULL || filename == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Check type and location. */
    type = oskar_telescope_model_type(telescope);
    location = oskar_telescope_model_location(telescope);
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
        return OSKAR_ERR_BAD_DATA_TYPE;
    if (location != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Check coordinate units are in metres. */
    if (telescope->coord_units != OSKAR_METRES)
        return OSKAR_ERR_BAD_UNITS;

    /* Open the file. */
    file = fopen(filename, "w");
    if (!file)
        return OSKAR_ERR_FILE_IO;

    /* Save the position of each station. */
    fprintf(file, "# Number of stations  = %i\n", telescope->num_stations);
    fprintf(file, "# Longitude [radians] = %f\n", telescope->longitude_rad);
    fprintf(file, "# Latitude [radians]  = %f\n", telescope->latitude_rad);
    fprintf(file, "# Altitude [metres]   = %f\n", telescope->altitude_m);
    fprintf(file, "# Local horizontal x(east), y(north), z(zenith) [metres]\n");
    if (type == OSKAR_SINGLE)
    {
        const float *x_hor, *y_hor, *z_hor;
        x_hor = (const float*)telescope->station_x_hor.data;
        y_hor = (const float*)telescope->station_y_hor.data;
        z_hor = (const float*)telescope->station_z_hor.data;
        for (i = 0; i < telescope->num_stations; ++i)
        {
            fprintf(file, "% 14.6f % 14.6f % 14.6f\n",
                    x_hor[i], y_hor[i], z_hor[i]);
        }
    }
    else if (type == OSKAR_DOUBLE)
    {
        const double *x_hor, *y_hor, *z_hor;
        x_hor = (const double*)telescope->station_x_hor.data;
        y_hor = (const double*)telescope->station_y_hor.data;
        z_hor = (const double*)telescope->station_z_hor.data;
        for (i = 0; i < telescope->num_stations; ++i)
        {
            fprintf(file, "% 14.6f % 14.6f % 14.6f\n",
                    x_hor[i], y_hor[i], z_hor[i]);
        }
    }

    /* Close the file. */
    fclose(file);

    return 0;
}

#ifdef __cplusplus
}
#endif
