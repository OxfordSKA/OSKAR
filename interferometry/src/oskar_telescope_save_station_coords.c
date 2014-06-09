/*
 * Copyright (c) 2012-2014, The University of Oxford
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
#include <oskar_telescope.h>

#include <oskar_telescope_save_station_coords.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_save_station_coords(
        const oskar_Telescope* telescope, const char* filename,
        int* status)
{
    int i, type, location, num_stations;
    FILE* file;

    /* Check all inputs. */
    if (!telescope || !filename || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check type and location. */
    type = oskar_telescope_precision(telescope);
    location = oskar_telescope_mem_location(telescope);
    num_stations = oskar_telescope_num_stations(telescope);
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (location != OSKAR_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Open the file. */
    file = fopen(filename, "w");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Save the position of each station. */
    fprintf(file, "# Number of stations  = %i\n", num_stations);
    fprintf(file, "# Longitude [radians] = %f\n", telescope->lon_rad);
    fprintf(file, "# Latitude [radians]  = %f\n", telescope->lat_rad);
    fprintf(file, "# Altitude [metres]   = %f\n", telescope->alt_metres);
    fprintf(file, "# Local horizontal x(east), y(north), z(up) [metres]\n");
    if (type == OSKAR_SINGLE)
    {
        const float *x_hor, *y_hor, *z_hor;
        x_hor = oskar_mem_float_const(
                oskar_telescope_station_true_x_enu_metres_const(telescope), status);
        y_hor = oskar_mem_float_const(
                oskar_telescope_station_true_y_enu_metres_const(telescope), status);
        z_hor = oskar_mem_float_const(
                oskar_telescope_station_true_z_enu_metres_const(telescope), status);
        for (i = 0; i < num_stations; ++i)
        {
            fprintf(file, "% 14.6f % 14.6f % 14.6f\n",
                    x_hor[i], y_hor[i], z_hor[i]);
        }
    }
    else if (type == OSKAR_DOUBLE)
    {
        const double *x_hor, *y_hor, *z_hor;
        x_hor = oskar_mem_double_const(
                oskar_telescope_station_true_x_enu_metres_const(telescope), status);
        y_hor = oskar_mem_double_const(
                oskar_telescope_station_true_y_enu_metres_const(telescope), status);
        z_hor = oskar_mem_double_const(
                oskar_telescope_station_true_z_enu_metres_const(telescope), status);
        for (i = 0; i < num_stations; ++i)
        {
            fprintf(file, "% 14.6f % 14.6f % 14.6f\n",
                    x_hor[i], y_hor[i], z_hor[i]);
        }
    }

    /* Close the file. */
    fclose(file);
}

#ifdef __cplusplus
}
#endif
