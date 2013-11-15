/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include <private_station.h>
#include <oskar_station.h>

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

void oskar_station_save_config(const char* filename,
        const oskar_Station* station, int* status)
{
    int i, location, type, num_elements;
    FILE* file;
    const oskar_Mem *x_weights, *y_weights, *z_weights;
    const oskar_Mem *x_signal, *y_signal, *z_signal;
    const oskar_Mem *gain, *gain_error, *phase, *phase_error, *weight;
    const oskar_Mem *cos_x, *cos_y, *sin_x, *sin_y;

    /* Check all inputs. */
    if (!filename || !station || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check type and location. */
    type = oskar_station_precision(station);
    location = oskar_station_location(station);
    num_elements = oskar_station_num_elements(station);
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (location != OSKAR_LOCATION_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Get pointers to the arrays. */
    x_weights = oskar_station_element_x_weights_const(station);
    y_weights = oskar_station_element_y_weights_const(station);
    z_weights = oskar_station_element_z_weights_const(station);
    x_signal = oskar_station_element_x_signal_const(station);
    y_signal = oskar_station_element_y_signal_const(station);
    z_signal = oskar_station_element_z_signal_const(station);
    gain = oskar_station_element_gain_const(station);
    gain_error = oskar_station_element_gain_error_const(station);
    phase = oskar_station_element_phase_offset_const(station);
    phase_error = oskar_station_element_phase_error_const(station);
    weight = oskar_station_element_weight_const(station);
    cos_x = oskar_station_element_cos_orientation_x_const(station);
    cos_y = oskar_station_element_cos_orientation_y_const(station);
    sin_x = oskar_station_element_sin_orientation_x_const(station);
    sin_y = oskar_station_element_sin_orientation_y_const(station);

    /* Open the file. */
    file = fopen(filename, "w");
    if (!file)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }

    /* Save the station data. */
    fprintf(file, "# Number of elements  = %i\n", num_elements);
    fprintf(file, "# Longitude [radians] = %f\n", station->longitude_rad);
    fprintf(file, "# Latitude [radians]  = %f\n", station->latitude_rad);
    fprintf(file, "# Altitude [metres]   = %f\n", station->altitude_m);
    fprintf(file, "# Local horizontal x(east), y(north), z(zenith) [metres], "
            "delta x, delta y, delta z [metres], gain, gain error, "
            "phase offset [deg], phase error [deg], weight(re), weight(im), "
            "X dipole azimuth [deg], Y dipole azimuth [deg]\n");
    if (type == OSKAR_DOUBLE)
    {
        const double *x_, *y_, *z_, *xs_, *ys_, *zs_;
        const double *gain_, *gain_error_, *phase_, *phase_error_;
        const double2 *weight_;
        const double *cos_x_, *cos_y_, *sin_x_, *sin_y_;
        x_ = oskar_mem_double_const(x_weights, status);
        y_ = oskar_mem_double_const(y_weights, status);
        z_ = oskar_mem_double_const(z_weights, status);
        xs_ = oskar_mem_double_const(x_signal, status);
        ys_ = oskar_mem_double_const(y_signal, status);
        zs_ = oskar_mem_double_const(z_signal, status);
        gain_ = oskar_mem_double_const(gain, status);
        gain_error_ = oskar_mem_double_const(gain_error, status);
        phase_ = oskar_mem_double_const(phase, status);
        phase_error_ = oskar_mem_double_const(phase_error, status);
        weight_ = oskar_mem_double2_const(weight, status);
        cos_x_ = oskar_mem_double_const(cos_x, status);
        cos_y_ = oskar_mem_double_const(cos_y, status);
        sin_x_ = oskar_mem_double_const(sin_x, status);
        sin_y_ = oskar_mem_double_const(sin_y, status);

        for (i = 0; i < num_elements; ++i)
        {
            double x, y, z;
            x = x_[i]; y = y_[i]; z = z_[i];
            fprintf(file, "% 14.6f % 14.6f % 14.6f % 14.6f % 14.6f % 14.6f "
                    "% 14.6f % 14.6f % 14.6f % 14.6f "
                    "% 14.6f % 14.6f % 14.6f % 14.6f\n",
                    x, y, z, (xs_[i] - x), (ys_[i] - y), (zs_[i] - z),
                    gain_[i], gain_error_[i], phase_[i], phase_error_[i],
                    weight_[i].x, weight_[i].y,
                    atan2(sin_x_[i], cos_x_[i]) * R2D,
                    atan2(sin_y_[i], cos_y_[i]) * R2D);
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        const float *x_, *y_, *z_, *xs_, *ys_, *zs_;
        const float *gain_, *gain_error_, *phase_, *phase_error_;
        const float2 *weight_;
        const float *cos_x_, *cos_y_, *sin_x_, *sin_y_;
        x_ = oskar_mem_float_const(x_weights, status);
        y_ = oskar_mem_float_const(y_weights, status);
        z_ = oskar_mem_float_const(z_weights, status);
        xs_ = oskar_mem_float_const(x_signal, status);
        ys_ = oskar_mem_float_const(y_signal, status);
        zs_ = oskar_mem_float_const(z_signal, status);
        gain_ = oskar_mem_float_const(gain, status);
        gain_error_ = oskar_mem_float_const(gain_error, status);
        phase_ = oskar_mem_float_const(phase, status);
        phase_error_ = oskar_mem_float_const(phase_error, status);
        weight_ = oskar_mem_float2_const(weight, status);
        cos_x_ = oskar_mem_float_const(cos_x, status);
        cos_y_ = oskar_mem_float_const(cos_y, status);
        sin_x_ = oskar_mem_float_const(sin_x, status);
        sin_y_ = oskar_mem_float_const(sin_y, status);

        for (i = 0; i < num_elements; ++i)
        {
            float x, y, z;
            x = x_[i]; y = y_[i]; z = z_[i];
            fprintf(file, "% 14.6f % 14.6f % 14.6f % 14.6f % 14.6f % 14.6f "
                    "% 14.6f % 14.6f % 14.6f % 14.6f "
                    "% 14.6f % 14.6f % 14.6f % 14.6f\n",
                    x, y, z, (xs_[i] - x), (ys_[i] - y), (zs_[i] - z),
                    gain_[i], gain_error_[i], phase_[i], phase_error_[i],
                    weight_[i].x, weight_[i].y,
                    atan2(sin_x_[i], cos_x_[i]) * R2D,
                    atan2(sin_y_[i], cos_y_[i]) * R2D);
        }
    }

    /* Close the file. */
    fclose(file);
}

#ifdef __cplusplus
}
#endif
