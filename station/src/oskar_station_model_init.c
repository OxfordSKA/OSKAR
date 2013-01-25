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

#include "station/oskar_element_model_init.h"
#include "station/oskar_station_model_init.h"
#include "station/oskar_system_noise_model_init.h"
#include "utility/oskar_mem_init.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_model_init(oskar_StationModel* model, int type, int location,
        int num_elements, int* status)
{
    /* Check all inputs. */
    if (!model || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check the type. */
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
        *status = OSKAR_ERR_BAD_DATA_TYPE;

    /* Initialise the memory. */
    oskar_mem_init(&model->x_signal, type, location, num_elements, 1, status);
    oskar_mem_init(&model->y_signal, type, location, num_elements, 1, status);
    oskar_mem_init(&model->z_signal, type, location, num_elements, 1, status);
    oskar_mem_init(&model->x_weights, type, location, num_elements, 1, status);
    oskar_mem_init(&model->y_weights, type, location, num_elements, 1, status);
    oskar_mem_init(&model->z_weights, type, location, num_elements, 1, status);
    oskar_mem_init(&model->weight, type | OSKAR_COMPLEX, location, num_elements, 1, status);
    oskar_mem_init(&model->gain, type, location, num_elements, 1, status);
    oskar_mem_init(&model->gain_error, type, location, num_elements, 1, status);
    oskar_mem_init(&model->phase_offset, type, location, num_elements, 1, status);
    oskar_mem_init(&model->phase_error, type, location, num_elements, 1, status);
    oskar_mem_init(&model->cos_orientation_x, type, location, num_elements, 1, status);
    oskar_mem_init(&model->sin_orientation_x, type, location, num_elements, 1, status);
    oskar_mem_init(&model->cos_orientation_y, type, location, num_elements, 1, status);
    oskar_mem_init(&model->sin_orientation_y, type, location, num_elements, 1, status);

    /* Initialise variables. */
    model->station_type = OSKAR_STATION_TYPE_AA;
    model->num_elements = num_elements;
    model->use_polarised_elements = OSKAR_TRUE;
    model->array_is_3d = OSKAR_FALSE;
    model->coord_units = OSKAR_METRES;
    model->apply_element_errors = OSKAR_FALSE;
    model->apply_element_weight = OSKAR_FALSE;
    model->single_element_model = OSKAR_TRUE;
    model->orientation_x = M_PI / 2.0;
    model->orientation_y = 0.0;
    model->gaussian_beam_fwhm_deg = 0.0;
    model->child = NULL;

    /* Allocate memory for the element model structure and initialise it. */
    model->element_pattern = (oskar_ElementModel*)
            malloc(sizeof(oskar_ElementModel));
    oskar_element_model_init(model->element_pattern, type, location, status);

    model->longitude_rad = 0.0;
    model->latitude_rad = 0.0;
    model->altitude_m = 0.0;
    model->beam_longitude_rad = 0.0;
    model->beam_latitude_rad = 0.0;
    model->beam_coord_type = 0;
    model->normalise_beam = OSKAR_FALSE;
    model->enable_array_pattern = OSKAR_TRUE;

    oskar_system_noise_model_init(&model->noise, type, location, status);
}

#ifdef __cplusplus
}
#endif
