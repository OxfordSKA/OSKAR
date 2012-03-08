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

#include "interferometry/oskar_telescope_model_analyse.h"
#include "interferometry/oskar_telescope_model_type.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "utility/oskar_mem_different.h"
#include "utility/oskar_mem_element_size.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef max
#define max(a, b) (((a) > (b)) ? (a) : (b))
#endif

void oskar_telescope_model_analyse(oskar_TelescopeModel* model)
{
    int i, j, finished_identical_station_check, num_stations, type;

    /* Set flags. */
    finished_identical_station_check = 0;
    model->identical_stations = 1;
    type = oskar_telescope_model_type(model);

    /* Find the maximum station size. */
    num_stations = model->num_stations;
    model->max_station_size = -INT_MAX;
    for (i = 0; i < num_stations; ++i)
    {
        model->max_station_size = max(model->station[i].num_elements,
                model->max_station_size);
    }

    /* Determine if element errors should be applied. */
    for (i = 0; i < num_stations; ++i)
    {
        oskar_StationModel* s;

        /* Get pointer to station. */
        s = &model->station[i];
        s->apply_element_errors = 0;

        if (type == OSKAR_DOUBLE)
        {
            double *amp, *amp_err, *phase, *phase_err;
            amp       = (double*)(s->gain.data);
            amp_err   = (double*)(s->gain_error.data);
            phase     = (double*)(s->phase_offset.data);
            phase_err = (double*)(s->phase_error.data);

            for (j = 0; j < s->num_elements; ++j)
            {
                if (amp[j] != 1.0 || phase[j] != 0.0)
                {
                    s->apply_element_errors = 1;
                }
                if (amp_err[j] != 0.0 || phase_err[j] != 0.0)
                {
                    s->apply_element_errors = 1;
                    model->identical_stations = 0;
                    finished_identical_station_check = 1;
                }
            }
        }
        else if (type == OSKAR_SINGLE)
        {
            float *amp, *amp_err, *phase, *phase_err;
            amp       = (float*)(s->gain.data);
            amp_err   = (float*)(s->gain_error.data);
            phase     = (float*)(s->phase_offset.data);
            phase_err = (float*)(s->phase_error.data);

            for (j = 0; j < s->num_elements; ++j)
            {
                if (amp[j] != 1.0f || phase[j] != 0.0)
                {
                    s->apply_element_errors = 1;
                }
                if (amp_err[j] != 0.0 || phase_err[j] != 0.0)
                {
                    s->apply_element_errors = 1;
                    model->identical_stations = 0;
                    finished_identical_station_check = 1;
                }
            }
        }
    }

    /* Check if the stations have different element positions, phase offsets
     * or gain factors. */
    if (!finished_identical_station_check)
    {
        oskar_Mem *x_weights0, *y_weights0, *z_weights0, *gain0, *phase0;
        oskar_Mem *x_signal0, *y_signal0, *z_signal0;
        int num_elements0;
        x_weights0 = &(model->station[0].x_weights);
        y_weights0 = &(model->station[0].y_weights);
        z_weights0 = &(model->station[0].z_weights);
        x_signal0 = &(model->station[0].x_signal);
        y_signal0 = &(model->station[0].y_signal);
        z_signal0 = &(model->station[0].z_signal);
        gain0 = &(model->station[0].gain);
        phase0 = &(model->station[0].phase_offset);
        num_elements0 = model->station[0].num_elements;

        for (i = 1; i < num_stations; ++i)
        {
            oskar_Mem *x_weights, *y_weights, *z_weights, *gain, *phase;
            oskar_Mem *x_signal, *y_signal, *z_signal;
            int num_elements;
            x_weights = &(model->station[i].x_weights);
            y_weights = &(model->station[i].y_weights);
            z_weights = &(model->station[i].z_weights);
            x_signal = &(model->station[i].x_signal);
            y_signal = &(model->station[i].y_signal);
            z_signal = &(model->station[i].z_signal);
            gain = &(model->station[i].gain);
            phase = &(model->station[i].phase_offset);
            num_elements = model->station[i].num_elements;

            /* Check if the number of elements is different. */
            if (num_elements != num_elements0)
            {
                model->identical_stations = 0;
                break;
            }

            /* Check if the memory contents are different. */
            if (oskar_mem_different(x_weights, x_weights0, num_elements))
            {
                model->identical_stations = 0;
                break;
            }
            if (oskar_mem_different(y_weights, y_weights0, num_elements))
            {
                model->identical_stations = 0;
                break;
            }
            if (oskar_mem_different(z_weights, z_weights0, num_elements))
            {
                model->identical_stations = 0;
                break;
            }
            if (oskar_mem_different(x_signal, x_signal0, num_elements))
            {
                model->identical_stations = 0;
                break;
            }
            if (oskar_mem_different(y_signal, y_signal0, num_elements))
            {
                model->identical_stations = 0;
                break;
            }
            if (oskar_mem_different(z_signal, z_signal0, num_elements))
            {
                model->identical_stations = 0;
                break;
            }
            if (oskar_mem_different(gain, gain0, num_elements))
            {
                model->identical_stations = 0;
                break;
            }
            if (oskar_mem_different(phase, phase0, num_elements))
            {
                model->identical_stations = 0;
                break;
            }
        }
    }

    /* Print summary data. */
    printf("\n");
    printf("= Telescope model\n");
    printf("  - Num. stations          = %u\n", model->num_stations);
    printf("  - Max station size       = %u\n", model->max_station_size);
    printf("  - Identical stations     = %s\n",
            model->identical_stations ? "true" : "false");
    printf("  - Use common sky         = %s\n",
            model->use_common_sky ? "true" : "false");
    printf("\n");
}

#ifdef __cplusplus
}
#endif
