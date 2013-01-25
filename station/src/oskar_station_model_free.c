/*
 * Copyright (c) 2013, The University of Oxford
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

#include "station/oskar_station_model_free.h"
#include "station/oskar_element_model_free.h"
#include "station/oskar_system_noise_model_free.h"
#include "utility/oskar_mem_free.h"
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_model_free(oskar_StationModel* model, int* status)
{
    /* Check all inputs. */
    if (!model || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Free the element data. */
    oskar_mem_free(&model->x_signal, status);
    oskar_mem_free(&model->y_signal, status);
    oskar_mem_free(&model->z_signal, status);
    oskar_mem_free(&model->x_weights, status);
    oskar_mem_free(&model->y_weights, status);
    oskar_mem_free(&model->z_weights, status);
    oskar_mem_free(&model->weight, status);
    oskar_mem_free(&model->gain, status);
    oskar_mem_free(&model->gain_error, status);
    oskar_mem_free(&model->phase_offset, status);
    oskar_mem_free(&model->phase_error, status);
    oskar_mem_free(&model->cos_orientation_x, status);
    oskar_mem_free(&model->sin_orientation_x, status);
    oskar_mem_free(&model->cos_orientation_y, status);
    oskar_mem_free(&model->sin_orientation_y, status);

    /* Free the element pattern data if it exists. */
    if (model->element_pattern)
    {
        oskar_element_model_free(model->element_pattern, status);

        /* Free the structure pointer. */
        free(model->element_pattern);
        model->element_pattern = NULL;
    }

    /* Initialise variables. */
    /* Do NOT set child or num_elements to 0 yet! */
    model->station_type = OSKAR_STATION_TYPE_AA;
    model->use_polarised_elements = OSKAR_TRUE;
    model->array_is_3d = OSKAR_FALSE;
    model->coord_units = OSKAR_METRES;
    model->apply_element_errors = OSKAR_FALSE;
    model->apply_element_weight = OSKAR_FALSE;
    model->single_element_model = OSKAR_TRUE;
    model->orientation_x = M_PI / 2.0;
    model->orientation_y = 0.0;
    model->element_pattern = NULL;

    model->longitude_rad = 0.0;
    model->latitude_rad = 0.0;
    model->altitude_m = 0.0;
    model->beam_longitude_rad = 0.0;
    model->beam_latitude_rad = 0.0;
    model->beam_coord_type = 0;
    model->normalise_beam = OSKAR_FALSE;
    model->enable_array_pattern = OSKAR_TRUE;

    /* Recursively free the child stations. */
    if (model->child)
    {
        int i = 0;
        for (i = 0; i < model->num_elements; ++i)
        {
            oskar_station_model_free(&(model->child[i]), status);
        }

        /* Free the array. */
        free(model->child);
        model->child = NULL;
    }

    /* Free the noise model. */
    oskar_system_noise_model_free(&model->noise, status);
}

#ifdef __cplusplus
}
#endif
