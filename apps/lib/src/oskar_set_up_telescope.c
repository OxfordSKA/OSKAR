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

#include "apps/lib/oskar_set_up_telescope.h"
#include "apps/lib/oskar_telescope_model_load.h"
#include "apps/lib/oskar_telescope_model_save.h"
#include "math/oskar_random_gaussian.h"
#include "interferometry/oskar_telescope_model_init.h"
#include "interferometry/oskar_telescope_model_analyse.h"
#include "station/oskar_evaluate_station_receiver_noise_stddev.h"
#include "station/oskar_station_model_type.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_free.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_set_value_real.h"

#include <math.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_set_up_telescope(oskar_TelescopeModel *telescope,
        const oskar_Settings* settings)
{
    int err, type, i, j;

    /* Initialise the structure in CPU memory. */
    type = settings->sim.double_precision ? OSKAR_DOUBLE : OSKAR_SINGLE;
    oskar_telescope_model_init(telescope, type, OSKAR_LOCATION_CPU, 0);

    /* Load the telescope configuration directory. */
    err = oskar_telescope_model_load(telescope, &settings->telescope);
    if (err)
    {
        fprintf(stderr, "== ERROR: Failed to load telescope configuration "
                "(%s).\n", oskar_get_error_string(err));
        return err;
    }

    /* Set phase centre. */
    telescope->ra0_rad = settings->obs.ra0_rad;
    telescope->dec0_rad = settings->obs.dec0_rad;

    /* Set other telescope parameters. */
    telescope->use_common_sky = settings->telescope.use_common_sky;
    telescope->bandwidth_hz = settings->obs.channel_bandwidth_hz;
    telescope->wavelength_metres = 0.0; /* This is set on a per-channel basis. */
    telescope->seed_time_variable_errors =
            settings->telescope.station.element.seed_time_variable_errors;

    /* Set other station parameters. */
    for (i = 0; i < telescope->num_stations; ++i)
    {
        telescope->station[i].ra0_rad = telescope->ra0_rad;
        telescope->station[i].dec0_rad = telescope->dec0_rad;
        telescope->station[i].station_type =
                settings->telescope.station.station_type;
        telescope->station[i].use_polarised_elements =
                settings->telescope.station.use_polarised_elements;
        telescope->station[i].evaluate_array_factor =
                settings->telescope.station.evaluate_array_factor;
        telescope->station[i].evaluate_element_factor =
                settings->telescope.station.evaluate_element_factor;
        telescope->station[i].normalise_beam =
                settings->telescope.station.normalise_beam;
    }

    /* Override station element systematic/fixed gain errors if required. */
    if (settings->telescope.station.element.gain > 0.0 ||
            settings->telescope.station.element.gain_error_fixed > 0.0)
    {
        double g, g_err;
        g = settings->telescope.station.element.gain;
        g_err = settings->telescope.station.element.gain_error_fixed;
        if (g <= 0.0) g = 1.0;
        srand(settings->telescope.station.element.seed_gain_errors);
        for (i = 0; i < telescope->num_stations; ++i)
        {
            int num_elements;
            num_elements = telescope->station[i].num_elements;
            type = oskar_station_model_type(&telescope->station[i]);
            if (type == OSKAR_DOUBLE)
            {
                double *gain;
                gain = (double*)(telescope->station[i].gain.data);
                for (j = 0; j < num_elements; ++j)
                    gain[j] = g + g_err * oskar_random_gaussian(0);
            }
            else if (type == OSKAR_SINGLE)
            {
                float *gain;
                gain = (float*)(telescope->station[i].gain.data);
                for (j = 0; j < num_elements; ++j)
                    gain[j] = g + g_err * oskar_random_gaussian(0);
            }
        }
    }

    /* Override station element time-variable gain errors if required. */
    if (settings->telescope.station.element.gain_error_time > 0.0)
    {
        for (i = 0; i < telescope->num_stations; ++i)
        {
            oskar_mem_set_value_real(&telescope->station[i].gain_error,
                    settings->telescope.station.element.gain_error_time);
        }
    }

    /* Override station element systematic/fixed phase errors if required. */
    if (settings->telescope.station.element.phase_error_fixed_rad > 0.0)
    {
        double p_err;
        p_err = settings->telescope.station.element.phase_error_fixed_rad;
        srand(settings->telescope.station.element.seed_phase_errors);
        for (i = 0; i < telescope->num_stations; ++i)
        {
            int num_elements;
            num_elements = telescope->station[i].num_elements;
            type = oskar_station_model_type(&telescope->station[i]);
            if (type == OSKAR_DOUBLE)
            {
                double *phase;
                phase = (double*)(telescope->station[i].phase_offset.data);
                for (j = 0; j < num_elements; ++j)
                    phase[j] = p_err * oskar_random_gaussian(0);
            }
            else if (type == OSKAR_SINGLE)
            {
                float *phase;
                phase = (float*)(telescope->station[i].phase_offset.data);
                for (j = 0; j < num_elements; ++j)
                    phase[j] = p_err * oskar_random_gaussian(0);
            }
        }
    }

    /* Override station element time-variable phase errors if required. */
    if (settings->telescope.station.element.phase_error_time_rad > 0.0)
    {
        for (i = 0; i < telescope->num_stations; ++i)
        {
            oskar_mem_set_value_real(&telescope->station[i].phase_error,
                    settings->telescope.station.element.phase_error_time_rad);
        }
    }

    /* Override station element position errors if required. */
    if (settings->telescope.station.element.position_error_xy_m > 0.0)
    {
        double p_err, delta_x, delta_y;
        p_err = settings->telescope.station.element.position_error_xy_m;
        srand(settings->telescope.station.element.seed_position_xy_errors);
        for (i = 0; i < telescope->num_stations; ++i)
        {
            int num_elements;
            type = oskar_station_model_type(&telescope->station[i]);
            num_elements = telescope->station[i].num_elements;
            if (type == OSKAR_DOUBLE)
            {
                double *xs, *ys, *xw, *yw;
                xs = (double*)(telescope->station[i].x_signal.data);
                ys = (double*)(telescope->station[i].y_signal.data);
                xw = (double*)(telescope->station[i].x_weights.data);
                yw = (double*)(telescope->station[i].y_weights.data);
                for (j = 0; j < num_elements; ++j)
                {
                    /* Generate random numbers from Gaussian distribution. */
                    delta_x = oskar_random_gaussian(&delta_y);
                    delta_x *= p_err;
                    delta_y *= p_err;
                    xs[j] = xw[j] + delta_x;
                    ys[j] = yw[j] + delta_y;
                }
            }
            else if (type == OSKAR_SINGLE)
            {
                float *xs, *ys, *xw, *yw;
                xs = (float*)(telescope->station[i].x_signal.data);
                ys = (float*)(telescope->station[i].y_signal.data);
                xw = (float*)(telescope->station[i].x_weights.data);
                yw = (float*)(telescope->station[i].y_weights.data);
                for (j = 0; j < num_elements; ++j)
                {
                    /* Generate random numbers from Gaussian distribution. */
                    delta_x = oskar_random_gaussian(&delta_y);
                    delta_x *= p_err;
                    delta_y *= p_err;
                    xs[j] = xw[j] + delta_x;
                    ys[j] = yw[j] + delta_y;
                }
            }
        }
    }

    /* Add variation to x-dipole orientations if required. */
    if (settings->telescope.station.element.x_orientation_error_rad > 0.0)
    {
        double p_err;
        p_err = settings->telescope.station.element.x_orientation_error_rad;
        srand(settings->telescope.station.element.seed_x_orientation_error);
        for (i = 0; i < telescope->num_stations; ++i)
        {
            int num_elements;
            double delta, angle;
            type = oskar_station_model_type(&telescope->station[i]);
            num_elements = telescope->station[i].num_elements;
            if (type == OSKAR_DOUBLE)
            {
                double *cos_x, *sin_x;
                cos_x = (double*)(telescope->station[i].cos_orientation_x.data);
                sin_x = (double*)(telescope->station[i].sin_orientation_x.data);
                for (j = 0; j < num_elements; ++j)
                {
                    /* Generate random number from Gaussian distribution. */
                    delta = p_err * oskar_random_gaussian(0);

                    /* Get the new angle. */
                    angle = delta + atan2(sin_x[j], cos_x[j]);
                    cos_x[j] = cos(angle);
                    sin_x[j] = sin(angle);
                }
            }
            else if (type == OSKAR_SINGLE)
            {
                float *cos_x, *sin_x;
                cos_x = (float*)(telescope->station[i].cos_orientation_x.data);
                sin_x = (float*)(telescope->station[i].sin_orientation_x.data);
                for (j = 0; j < num_elements; ++j)
                {
                    /* Generate random number from Gaussian distribution. */
                    delta = p_err * oskar_random_gaussian(0);

                    /* Get the new angle. */
                    angle = delta + atan2(sin_x[j], cos_x[j]);
                    cos_x[j] = (float) cos(angle);
                    sin_x[j] = (float) sin(angle);
                }
            }
        }
    }

    /* Add variation to y-dipole orientations if required. */
    if (settings->telescope.station.element.y_orientation_error_rad > 0.0)
    {
        double p_err;
        p_err = settings->telescope.station.element.y_orientation_error_rad;
        srand(settings->telescope.station.element.seed_y_orientation_error);
        for (i = 0; i < telescope->num_stations; ++i)
        {
            int num_elements;
            double delta, angle;
            type = oskar_station_model_type(&telescope->station[i]);
            num_elements = telescope->station[i].num_elements;
            if (type == OSKAR_DOUBLE)
            {
                double *cos_y, *sin_y;
                cos_y = (double*)(telescope->station[i].cos_orientation_y.data);
                sin_y = (double*)(telescope->station[i].sin_orientation_y.data);
                for (j = 0; j < num_elements; ++j)
                {
                    /* Generate random number from Gaussian distribution. */
                    delta = p_err * oskar_random_gaussian(0);

                    /* Get the new angle. */
                    angle = delta + atan2(sin_y[j], cos_y[j]);
                    cos_y[j] = cos(angle);
                    sin_y[j] = sin(angle);
                }
            }
            else if (type == OSKAR_SINGLE)
            {
                float *cos_y, *sin_y;
                cos_y = (float*)(telescope->station[i].cos_orientation_y.data);
                sin_y = (float*)(telescope->station[i].sin_orientation_y.data);
                for (j = 0; j < num_elements; ++j)
                {
                    /* Generate random number from Gaussian distribution. */
                    delta = p_err * oskar_random_gaussian(0);

                    /* Get the new angle. */
                    angle = delta + atan2(sin_y[j], cos_y[j]);
                    cos_y[j] = (float) cos(angle);
                    sin_y[j] = (float) sin(angle);
                }
            }
        }
    }

    /* Analyse telescope model to determine whether stations are identical,
     * whether to apply element errors and/or weights. */
    oskar_telescope_model_analyse(telescope);

    /* Save the telescope configuration in a new directory if required. */
    if (settings->telescope.output_config_directory)
    {
        /* Check that the input and output directories are different. */
        if (strcmp(settings->telescope.output_config_directory,
                settings->telescope.config_directory))
        {
            err = oskar_telescope_model_save(telescope,
                    settings->telescope.output_config_directory);
            if (err)
            {
                fprintf(stderr, "== ERROR: Failed to save telescope "
                        "configuration (%s).\n", oskar_get_error_string(err));
                return err;
            }
        }
        else
        {
            printf("== WARNING: Will not overwrite input "
                    "telescope directory!\n");
        }
    }

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
