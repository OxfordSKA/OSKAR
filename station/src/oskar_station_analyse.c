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
    double *x_alpha, *x_beta, *x_gamma, *y_alpha, *y_beta, *y_gamma;
    char* mount_type;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check station model is in CPU-accessible memory. */
    if (oskar_station_mem_location(station) != OSKAR_CPU)
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
    station->common_element_orientation = 1;

    /* Analyse orientations separately (always double precision). */
    mount_type = oskar_mem_char(station->element_mount_types_cpu);
    x_alpha = oskar_mem_double(station->element_x_alpha_cpu, status);
    x_beta  = oskar_mem_double(station->element_x_beta_cpu, status);
    x_gamma = oskar_mem_double(station->element_x_gamma_cpu, status);
    y_alpha = oskar_mem_double(station->element_y_alpha_cpu, status);
    y_beta  = oskar_mem_double(station->element_y_beta_cpu, status);
    y_gamma = oskar_mem_double(station->element_y_gamma_cpu, status);
    for (i = 1; i < station->num_elements; ++i)
    {
        if (mount_type[i] != mount_type[0] ||
                x_alpha[i] != x_alpha[0] ||
                x_beta[i] != x_beta[0] ||
                x_gamma[i] != x_gamma[0] ||
                y_alpha[i] != y_alpha[0] ||
                y_beta[i] != y_beta[0] ||
                y_gamma[i] != y_gamma[0])
        {
            station->common_element_orientation = 0;
            break;
        }
    }

    if (type == OSKAR_DOUBLE)
    {
        double *z_true, *z_meas, *amp, *amp_err, *phase, *phase_err;
        double2 *weights;
        z_true    = oskar_mem_double(station->element_true_z_enu_metres, status);
        z_meas    = oskar_mem_double(station->element_measured_z_enu_metres, status);
        amp       = oskar_mem_double(station->element_gain, status);
        amp_err   = oskar_mem_double(station->element_gain_error, status);
        phase     = oskar_mem_double(station->element_phase_offset_rad, status);
        phase_err = oskar_mem_double(station->element_phase_error_rad, status);
        weights   = oskar_mem_double2(station->element_weight, status);

        for (i = 0; i < station->num_elements; ++i)
        {
            if (z_true[i] != 0.0 || z_meas[i] != 0.0)
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
        }
    }
    else if (type == OSKAR_SINGLE)
    {
        float *z_true, *z_meas, *amp, *amp_err, *phase, *phase_err;
        float2 *weights;
        z_true    = oskar_mem_float(station->element_true_z_enu_metres, status);
        z_meas    = oskar_mem_float(station->element_measured_z_enu_metres, status);
        amp       = oskar_mem_float(station->element_gain, status);
        amp_err   = oskar_mem_float(station->element_gain_error, status);
        phase     = oskar_mem_float(station->element_phase_offset_rad, status);
        phase_err = oskar_mem_float(station->element_phase_error_rad, status);
        weights   = oskar_mem_float2(station->element_weight, status);

        for (i = 0; i < station->num_elements; ++i)
        {
            if (z_true[i] != 0.0 || z_meas[i] != 0.0)
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
        }
    }

    /* Check if station has child stations. */
    if (oskar_station_has_child(station))
    {
        /* Recursively analyse all child stations. */
        for (i = 0; i < station->num_elements; ++i)
        {
            oskar_station_analyse(oskar_station_child(station, i),
                    finished_identical_station_check, status);
        }

        /* Check if we need to examine every station. */
        if (*finished_identical_station_check)
        {
            station->identical_children = 0;
        }
        else
        {
            /* Check if child stations are identical. */
            station->identical_children = 1;
            for (i = 1; i < station->num_elements; ++i)
            {
                if (oskar_station_different(
                        oskar_station_child_const(station, 0),
                        oskar_station_child_const(station, i), status))
                {
                    station->identical_children = 0;
                }
            }
        }
    }
}

#ifdef __cplusplus
}
#endif
