/*
 * Copyright (c) 2014-2015, The University of Oxford
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

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_save_layout(const char* filename,
        const oskar_Station* station, int* status)
{
    int i, location, type, num_elements;
    FILE* file;
    const oskar_Mem *x_weights, *y_weights, *z_weights;
    const oskar_Mem *x_signal, *y_signal, *z_signal;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check type and location. */
    type = oskar_station_precision(station);
    location = oskar_station_mem_location(station);
    num_elements = oskar_station_num_elements(station);
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

    /* Get pointers to the arrays. */
    x_weights = oskar_station_element_measured_x_enu_metres_const(station);
    y_weights = oskar_station_element_measured_y_enu_metres_const(station);
    z_weights = oskar_station_element_measured_z_enu_metres_const(station);
    x_signal = oskar_station_element_true_x_enu_metres_const(station);
    y_signal = oskar_station_element_true_y_enu_metres_const(station);
    z_signal = oskar_station_element_true_z_enu_metres_const(station);

    /* Open the file. */
    file = fopen(filename, "w");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Save the station data. */
    fprintf(file, "# Number of elements  = %i\n", num_elements);
    fprintf(file, "# Longitude [radians] = %f\n", station->lon_rad);
    fprintf(file, "# Latitude [radians]  = %f\n", station->lat_rad);
    fprintf(file, "# Altitude [metres]   = %f\n", station->alt_metres);
    fprintf(file, "# Local horizontal x (east), y (north), z (up) [metres], "
            "delta x, delta y, delta z [metres]\n");
    if (type == OSKAR_DOUBLE)
    {
        const double *x_, *y_, *z_, *xs_, *ys_, *zs_;
        x_ = oskar_mem_double_const(x_weights, status);
        y_ = oskar_mem_double_const(y_weights, status);
        z_ = oskar_mem_double_const(z_weights, status);
        xs_ = oskar_mem_double_const(x_signal, status);
        ys_ = oskar_mem_double_const(y_signal, status);
        zs_ = oskar_mem_double_const(z_signal, status);

        for (i = 0; i < num_elements; ++i)
        {
            double x, y, z;
            x = x_[i]; y = y_[i]; z = z_[i];
            fprintf(file, "% 14.6f % 14.6f % 14.6f % 14.6f % 14.6f % 14.6f\n",
                    x, y, z, (xs_[i] - x), (ys_[i] - y), (zs_[i] - z));
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        const float *x_, *y_, *z_, *xs_, *ys_, *zs_;
        x_ = oskar_mem_float_const(x_weights, status);
        y_ = oskar_mem_float_const(y_weights, status);
        z_ = oskar_mem_float_const(z_weights, status);
        xs_ = oskar_mem_float_const(x_signal, status);
        ys_ = oskar_mem_float_const(y_signal, status);
        zs_ = oskar_mem_float_const(z_signal, status);

        for (i = 0; i < num_elements; ++i)
        {
            float x, y, z;
            x = x_[i]; y = y_[i]; z = z_[i];
            fprintf(file, "% 14.6f % 14.6f % 14.6f % 14.6f % 14.6f % 14.6f\n",
                    x, y, z, (xs_[i] - x), (ys_[i] - y), (zs_[i] - z));
        }
    }

    /* Close the file. */
    fclose(file);
}

#ifdef __cplusplus
}
#endif
