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
#include "station/oskar_station_model_analyse.h"
#include "utility/oskar_mem_different.h"
#include "utility/oskar_mem_element_size.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#include <vector_types.h>

#ifdef __cplusplus
extern "C" {
#endif

#ifndef MAX
#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

void oskar_telescope_model_analyse(oskar_TelescopeModel* model)
{
    int i, finished_identical_station_check, num_stations;

    /* Set flags. */
    finished_identical_station_check = 0;
    model->identical_stations = 1;

    /* Find the maximum station size. */
    num_stations = model->num_stations;
    model->max_station_size = -INT_MAX;
    for (i = 0; i < num_stations; ++i)
    {
        model->max_station_size = MAX(model->station[i].num_elements,
                model->max_station_size);
    }

    /* Analyse each station. */
    for (i = 0; i < num_stations; ++i)
    {
        oskar_StationModel* s;
        s = &model->station[i];
        oskar_station_model_analyse(s, &finished_identical_station_check);
    }

    /* Check if the stations have different element positions, phase offsets
     * or gain factors. */
    if (finished_identical_station_check)
    {
        model->identical_stations = 0;
    }
    else
    {
        oskar_Mem *x_weights0, *y_weights0, *z_weights0, *gain0, *phase0;
        oskar_Mem *x_signal0, *y_signal0, *z_signal0;
        oskar_Mem *cos_x0, *sin_x0, *cos_y0, *sin_y0;
        oskar_Mem *weights0;
        int station_type0, num_elements0, element_type0, array_is_3d0;
        int apply_element_errors0, apply_element_weight0, single_element_model0;
        x_weights0 = &(model->station[0].x_weights);
        y_weights0 = &(model->station[0].y_weights);
        z_weights0 = &(model->station[0].z_weights);
        x_signal0 = &(model->station[0].x_signal);
        y_signal0 = &(model->station[0].y_signal);
        z_signal0 = &(model->station[0].z_signal);
        gain0 = &(model->station[0].gain);
        phase0 = &(model->station[0].phase_offset);
        weights0 = &(model->station[0].weight);
        cos_x0 = &(model->station[0].cos_orientation_x);
        sin_x0 = &(model->station[0].sin_orientation_x);
        cos_y0 = &(model->station[0].cos_orientation_y);
        sin_y0 = &(model->station[0].sin_orientation_y);
        station_type0 = model->station[0].station_type;
        num_elements0 = model->station[0].num_elements;
        element_type0 = model->station[0].element_type;
        array_is_3d0 = model->station[0].array_is_3d;
        apply_element_errors0 = model->station[0].apply_element_errors;
        apply_element_weight0 = model->station[0].apply_element_weight;
        single_element_model0 = model->station[0].single_element_model;

        for (i = 1; i < num_stations; ++i)
        {
            oskar_Mem *x_weights, *y_weights, *z_weights, *gain, *phase;
            oskar_Mem *x_signal, *y_signal, *z_signal;
            oskar_Mem *cos_x, *sin_x, *cos_y, *sin_y;
            oskar_Mem *weights;
            int station_type, num_elements, element_type, array_is_3d;
            int apply_element_errors, apply_element_weight, single_element_model;
            x_weights = &(model->station[i].x_weights);
            y_weights = &(model->station[i].y_weights);
            z_weights = &(model->station[i].z_weights);
            x_signal = &(model->station[i].x_signal);
            y_signal = &(model->station[i].y_signal);
            z_signal = &(model->station[i].z_signal);
            gain = &(model->station[i].gain);
            phase = &(model->station[i].phase_offset);
            weights = &(model->station[i].weight);
            cos_x = &(model->station[i].cos_orientation_x);
            sin_x = &(model->station[i].sin_orientation_x);
            cos_y = &(model->station[i].cos_orientation_y);
            sin_y = &(model->station[i].sin_orientation_y);
            station_type = model->station[i].station_type;
            num_elements = model->station[i].num_elements;
            element_type = model->station[i].element_type;
            array_is_3d = model->station[i].array_is_3d;
            apply_element_errors = model->station[i].apply_element_errors;
            apply_element_weight = model->station[i].apply_element_weight;
            single_element_model = model->station[i].single_element_model;

            /* Check if the meta-data are different. */
            if (station_type != station_type0 ||
                    num_elements != num_elements0 ||
                    element_type != element_type0 ||
                    array_is_3d != array_is_3d0 ||
                    apply_element_errors != apply_element_errors0 ||
                    apply_element_weight != apply_element_weight0 ||
                    single_element_model != single_element_model0)
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
            if (oskar_mem_different(weights, weights0, num_elements))
            {
                model->identical_stations = 0;
                break;
            }
            if (oskar_mem_different(cos_x, cos_x0, num_elements))
            {
                model->identical_stations = 0;
                break;
            }
            if (oskar_mem_different(sin_x, sin_x0, num_elements))
            {
                model->identical_stations = 0;
                break;
            }
            if (oskar_mem_different(cos_y, cos_y0, num_elements))
            {
                model->identical_stations = 0;
                break;
            }
            if (oskar_mem_different(sin_y, sin_y0, num_elements))
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
