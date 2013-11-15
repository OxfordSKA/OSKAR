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

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_analyse(oskar_Station* station,
        int* finished_identical_station_check, int* status)
{
    int i, type;

    /* Check all inputs. */
    if (!station || !finished_identical_station_check || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check station model is in CPU-accessible memory. */
    if (oskar_station_location(station) != OSKAR_LOCATION_CPU)
    {
        *status = OSKAR_ERR_BAD_LOCATION;
        return;
    }

    /* Get type. */
    type = oskar_station_precision(station);

    /* Set default station flags. */
    station->array_is_3d = 0;
    station->apply_element_errors = 0;
    station->apply_element_weight = 0;
    station->single_element_model = 1;

    if (type == OSKAR_DOUBLE)
    {
        double *z_signal, *z_weights, *amp, *amp_err, *phase, *phase_err;
        double *cos_x, *sin_x, *cos_y, *sin_y;
        double2 *weights;
        z_signal  = oskar_mem_double(&station->z_signal, status);
        z_weights = oskar_mem_double(&station->z_weights, status);
        amp       = oskar_mem_double(&station->gain, status);
        amp_err   = oskar_mem_double(&station->gain_error, status);
        phase     = oskar_mem_double(&station->phase_offset, status);
        phase_err = oskar_mem_double(&station->phase_error, status);
        cos_x     = oskar_mem_double(&station->cos_orientation_x, status);
        sin_x     = oskar_mem_double(&station->sin_orientation_x, status);
        cos_y     = oskar_mem_double(&station->cos_orientation_y, status);
        sin_y     = oskar_mem_double(&station->sin_orientation_y, status);
        weights   = oskar_mem_double2(&station->weight, status);

        for (i = 0; i < station->num_elements; ++i)
        {
            if (z_signal[i] != 0.0 || z_weights[i] != 0.0)
            {
                station->array_is_3d = 1;
            }
            if (amp[i] != 1.0 || phase[i] != 0.0)
            {
                station->apply_element_errors = 1;
            }
            if (amp_err[i] != 0.0 || phase_err[i] != 0.0)
            {
                station->apply_element_errors = 1;
                *finished_identical_station_check = 1;
            }
            if (weights[i].x != 1.0 || weights[i].y != 0.0)
            {
                station->apply_element_weight = 1;
            }
            if (cos_x[i] != cos_x[0] || sin_x[i] != sin_x[0] ||
                    cos_y[i] != cos_y[0] || sin_y[i] != sin_y[0])
            {
                station->single_element_model = 0;
            }
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        float *z_signal, *z_weights, *amp, *amp_err, *phase, *phase_err;
        float *cos_x, *sin_x, *cos_y, *sin_y;
        float2 *weights;
        z_signal  = oskar_mem_float(&station->z_signal, status);
        z_weights = oskar_mem_float(&station->z_weights, status);
        amp       = oskar_mem_float(&station->gain, status);
        amp_err   = oskar_mem_float(&station->gain_error, status);
        phase     = oskar_mem_float(&station->phase_offset, status);
        phase_err = oskar_mem_float(&station->phase_error, status);
        cos_x     = oskar_mem_float(&station->cos_orientation_x, status);
        sin_x     = oskar_mem_float(&station->sin_orientation_x, status);
        cos_y     = oskar_mem_float(&station->cos_orientation_y, status);
        sin_y     = oskar_mem_float(&station->sin_orientation_y, status);
        weights   = oskar_mem_float2(&station->weight, status);

        for (i = 0; i < station->num_elements; ++i)
        {
            if (z_signal[i] != 0.0 || z_weights[i] != 0.0)
            {
                station->array_is_3d = 1;
            }
            if (amp[i] != 1.0f || phase[i] != 0.0)
            {
                station->apply_element_errors = 1;
            }
            if (amp_err[i] != 0.0 || phase_err[i] != 0.0)
            {
                station->apply_element_errors = 1;
                *finished_identical_station_check = 1;
            }
            if (weights[i].x != 1.0f || weights[i].y != 0.0)
            {
                station->apply_element_weight = 1;
            }
            if (cos_x[i] != cos_x[0] || sin_x[i] != sin_x[0] ||
                    cos_y[i] != cos_y[0] || sin_y[i] != sin_y[0])
            {
                station->single_element_model = 0;
            }
        }
    }

    /* Recursively analyse all child stations. */
    if (oskar_station_has_child(station))
    {
        for (i = 0; i < station->num_elements; ++i)
        {
            oskar_station_analyse(oskar_station_child(station, i),
                    finished_identical_station_check, status);
        }
    }
}

#ifdef __cplusplus
}
#endif
