/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_save_layout(const oskar_Telescope* telescope,
        const char* filename, int* status)
{
    int i, type, location, num_stations;
    FILE* file;
    const oskar_Mem *x_weights, *y_weights, *z_weights;
    const oskar_Mem *x_signal, *y_signal, *z_signal;

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

    /* Get pointers to the arrays. */
    x_weights = oskar_telescope_station_measured_x_enu_metres_const(telescope);
    y_weights = oskar_telescope_station_measured_y_enu_metres_const(telescope);
    z_weights = oskar_telescope_station_measured_z_enu_metres_const(telescope);
    x_signal = oskar_telescope_station_true_x_enu_metres_const(telescope);
    y_signal = oskar_telescope_station_true_y_enu_metres_const(telescope);
    z_signal = oskar_telescope_station_true_z_enu_metres_const(telescope);

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

        for (i = 0; i < num_stations; ++i)
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

        for (i = 0; i < num_stations; ++i)
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
