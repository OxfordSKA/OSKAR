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

#include "station/oskar_station_model_free.h"
#include "station/oskar_element_model_free.h"
#include "utility/oskar_mem_free.h"
#include <stdlib.h>
#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_station_model_free(oskar_StationModel* model)
{
    int error = 0;

    /* Check for sane inputs. */
    if (model == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Free the element data. */
    error = oskar_mem_free(&model->x_signal);
    if (error) return error;
    error = oskar_mem_free(&model->y_signal);
    if (error) return error;
    error = oskar_mem_free(&model->z_signal);
    if (error) return error;
    error = oskar_mem_free(&model->x_weights);
    if (error) return error;
    error = oskar_mem_free(&model->y_weights);
    if (error) return error;
    error = oskar_mem_free(&model->z_weights);
    if (error) return error;
    error = oskar_mem_free(&model->weight);
    if (error) return error;
    error = oskar_mem_free(&model->gain);
    if (error) return error;
    error = oskar_mem_free(&model->gain_error);
    if (error) return error;
    error = oskar_mem_free(&model->phase_offset);
    if (error) return error;
    error = oskar_mem_free(&model->phase_error);
    if (error) return error;
    error = oskar_mem_free(&model->cos_orientation_x);
    if (error) return error;
    error = oskar_mem_free(&model->sin_orientation_x);
    if (error) return error;
    error = oskar_mem_free(&model->cos_orientation_y);
    if (error) return error;
    error = oskar_mem_free(&model->sin_orientation_y);
    if (error) return error;

    /* Free the receiver noise data. */
    error = oskar_mem_free(&model->total_receiver_noise);
    if (error) return error;

    /* Free the element pattern data if it exists. */
    if (model->element_pattern)
    {
        error = oskar_element_model_free(model->element_pattern);
        if (error) return error;

        /* Free the structure pointer. */
        free(model->element_pattern);
        model->element_pattern = NULL;
    }

    /* Initialise variables. */
    /* Do NOT set child or num_elements to 0 yet! */
    model->station_type = OSKAR_STATION_TYPE_AA;
    model->element_type = OSKAR_STATION_ELEMENT_TYPE_POINT;
    model->array_is_3d = OSKAR_FALSE;
    model->coord_units = OSKAR_METRES;
    model->apply_element_errors = OSKAR_FALSE;
    model->apply_element_weight = OSKAR_FALSE;
    model->single_element_model = OSKAR_TRUE;
    model->orientation_x = M_PI / 2.0;
    model->orientation_y = 0.0;

    model->parent = NULL;
    model->element_pattern = NULL;

    model->longitude_rad = 0.0;
    model->latitude_rad = 0.0;
    model->altitude_m = 0.0;
    model->ra0_rad = 0.0;
    model->dec0_rad = 0.0;
    model->normalise_beam = OSKAR_FALSE;
    model->evaluate_array_factor = OSKAR_TRUE;
    model->evaluate_element_factor = OSKAR_TRUE;
    model->bit_depth = 0;

    /* Recursively free the child stations. */
    if (model->child)
    {
        int i = 0;
        for (i = 0; i < model->num_elements; ++i)
        {
            error = oskar_station_model_free(&(model->child[i]));
            if (error) return error;
        }

        /* Free the array. */
        free(model->child);
        model->child = NULL;
    }

    return 0;
}

#ifdef __cplusplus
}
#endif
