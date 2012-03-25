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

#include "station/oskar_station_model_location.h"
#include "station/oskar_station_model_type.h"
#include "station/oskar_station_model_save_config.h"
#include "utility/oskar_vector_types.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define R2D (180.0 / M_PI)

int oskar_station_model_save_config(const char* filename,
        const oskar_StationModel* station)
{
    int i, location, type;
    FILE* file;

    /* Sanity check on inputs. */
    if (filename == NULL || station == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Check type and location. */
    type = oskar_station_model_type(station);
    location = oskar_station_model_location(station);
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
        return OSKAR_ERR_BAD_DATA_TYPE;
    if (location != OSKAR_LOCATION_CPU)
        return OSKAR_ERR_BAD_LOCATION;

    /* Check coordinate units are in metres. */
    if (station->coord_units != OSKAR_METRES)
        return OSKAR_ERR_BAD_UNITS;

    /* Open the file. */
    file = fopen(filename, "w");
    if (!file)
        return OSKAR_ERR_FILE_IO;

    /* Save the station data. */
    fprintf(file, "# Number of elements  = %i\n", station->num_elements);
    fprintf(file, "# Longitude [radians] = %f\n", station->longitude_rad);
    fprintf(file, "# Latitude [radians]  = %f\n", station->latitude_rad);
    fprintf(file, "# Altitude [metres]   = %f\n", station->altitude_m);
    fprintf(file, "# Local horizontal x(east), y(north), z(zenith) [metres], "
            "delta x, delta y, delta z [metres], gain, gain error, "
            "phase offset [deg], phase error [deg], weight(re), weight(im), "
            "X dipole azimuth [deg], Y dipole azimuth [deg]\n");
    if (type == OSKAR_DOUBLE)
    {
        double x, y, z, delta_x, delta_y, delta_z;
        double gain, gain_error, phase_offset, phase_error;
        double weight_re, weight_im, x_azimuth, y_azimuth;
        for (i = 0; i < station->num_elements; ++i)
        {
            x = ((double*)station->x_weights.data)[i];
            y = ((double*)station->y_weights.data)[i];
            z = ((double*)station->z_weights.data)[i];
            delta_x = ((double*)station->x_signal.data)[i] - x;
            delta_y = ((double*)station->y_signal.data)[i] - y;
            delta_z = ((double*)station->z_signal.data)[i] - z;
            gain = ((double*)station->gain.data)[i];
            gain_error = ((double*)station->gain_error.data)[i];
            phase_offset = ((double*)station->phase_offset.data)[i] * R2D;
            phase_error = ((double*)station->phase_error.data)[i] * R2D;
            weight_re = ((double2*)station->weight.data)[i].x;
            weight_im = ((double2*)station->weight.data)[i].y;
            x_azimuth = atan2(((double*)station->sin_orientation_x.data)[i],
                    ((double*)station->cos_orientation_x.data)[i]) * R2D;
            y_azimuth = atan2(((double*)station->sin_orientation_y.data)[i],
                    ((double*)station->cos_orientation_y.data)[i]) * R2D;
            fprintf(file, "% -12.6f,% -12.6f,% -12.6f,% -12.6f,% -12.6f,"
                    "% -12.6f,% -12.6f,% -12.6f,% -12.6f,% -12.6f,"
                    "% -12.6f,% -12.6f,% -12.6f,% -12.6f\n",
                    x, y, z, delta_x, delta_y, delta_z, gain, gain_error,
                    phase_offset, phase_error, weight_re, weight_im,
                    x_azimuth, y_azimuth);
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        float x, y, z, delta_x, delta_y, delta_z;
        float gain, gain_error, phase_offset, phase_error;
        float weight_re, weight_im, x_azimuth, y_azimuth;
        for (i = 0; i < station->num_elements; ++i)
        {
            x = ((float*)station->x_weights.data)[i];
            y = ((float*)station->y_weights.data)[i];
            z = ((float*)station->z_weights.data)[i];
            delta_x = ((float*)station->x_signal.data)[i] - x;
            delta_y = ((float*)station->y_signal.data)[i] - y;
            delta_z = ((float*)station->z_signal.data)[i] - z;
            gain = ((float*)station->gain.data)[i];
            gain_error = ((float*)station->gain_error.data)[i];
            phase_offset = ((float*)station->phase_offset.data)[i] * R2D;
            phase_error = ((float*)station->phase_error.data)[i] * R2D;
            weight_re = ((float2*)station->weight.data)[i].x;
            weight_im = ((float2*)station->weight.data)[i].y;
            x_azimuth = atan2(((float*)station->sin_orientation_x.data)[i],
                    ((float*)station->cos_orientation_x.data)[i]) * R2D;
            y_azimuth = atan2(((float*)station->sin_orientation_y.data)[i],
                    ((float*)station->cos_orientation_y.data)[i]) * R2D;
            fprintf(file, "% -12.6f,% -12.6f,% -12.6f,% -12.6f,% -12.6f,"
                    "% -12.6f,% -12.6f,% -12.6f,% -12.6f,% -12.6f,"
                    "% -12.6f,% -12.6f,% -12.6f,% -12.6f\n",
                    x, y, z, delta_x, delta_y, delta_z, gain, gain_error,
                    phase_offset, phase_error, weight_re, weight_im,
                    x_azimuth, y_azimuth);
        }
    }

    /* Close the file. */
    fclose(file);

    return 0;
}

#ifdef __cplusplus
}
#endif
