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

#include "station/oskar_element_model_copy.h"
#include "station/oskar_element_model_init.h"
#include "station/oskar_element_model_type.h"
#include "station/oskar_station_model_copy.h"
#include "station/oskar_station_model_location.h"
#include "utility/oskar_mem_copy.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_station_model_copy(oskar_StationModel* dst,
        const oskar_StationModel* src)
{
    int error = 0;

    /* Check that all pointers are not NULL. */
    if (src == NULL || dst == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    /* Copy the memory blocks. */
    oskar_mem_copy(&dst->x_signal, &src->x_signal, &error);
    oskar_mem_copy(&dst->y_signal, &src->y_signal, &error);
    oskar_mem_copy(&dst->z_signal, &src->z_signal, &error);
    oskar_mem_copy(&dst->x_weights, &src->x_weights, &error);
    oskar_mem_copy(&dst->y_weights, &src->y_weights, &error);
    oskar_mem_copy(&dst->z_weights, &src->z_weights, &error);
    oskar_mem_copy(&dst->weight, &src->weight, &error);
    oskar_mem_copy(&dst->gain, &src->gain, &error);
    oskar_mem_copy(&dst->gain_error, &src->gain_error, &error);
    oskar_mem_copy(&dst->phase_offset, &src->phase_offset, &error);
    oskar_mem_copy(&dst->phase_error, &src->phase_error, &error);
    oskar_mem_copy(&dst->cos_orientation_x, &src->cos_orientation_x, &error);
    oskar_mem_copy(&dst->sin_orientation_x, &src->sin_orientation_x, &error);
    oskar_mem_copy(&dst->cos_orientation_y, &src->cos_orientation_y, &error);
    oskar_mem_copy(&dst->sin_orientation_y, &src->sin_orientation_y, &error);
    if (error) return error;

    /* Copy the meta data. */
    dst->station_type = src->station_type;
    dst->num_elements = src->num_elements;
    dst->use_polarised_elements = src->use_polarised_elements;
    dst->array_is_3d = src->array_is_3d;
    dst->coord_units = src->coord_units;
    dst->apply_element_errors = src->apply_element_errors;
    dst->apply_element_weight = src->apply_element_weight;
    dst->single_element_model = src->single_element_model;
    dst->orientation_x = src->orientation_x;
    dst->orientation_y = src->orientation_y;
    dst->longitude_rad = src->longitude_rad;
    dst->latitude_rad = src->latitude_rad;
    dst->altitude_m = src->altitude_m;
    dst->ra0_rad = src->ra0_rad;
    dst->dec0_rad = src->dec0_rad;
    dst->normalise_beam = src->normalise_beam;
    dst->evaluate_array_factor = src->evaluate_array_factor;
    dst->evaluate_element_factor = src->evaluate_element_factor;
    dst->bit_depth = src->bit_depth;

    /* TODO Work out how to deal with element pattern data properly! */
    if (src->element_pattern)
    {
        /* Initialise the element model. */
        error = oskar_element_model_init(dst->element_pattern,
                oskar_element_model_type(src->element_pattern),
                oskar_station_model_location(dst));
        if (error) return error;

        /* Copy the element model data. */
        oskar_element_model_copy(dst->element_pattern,
                src->element_pattern, &error);
        if (error) return error;
    }

    /* TODO Work out how to deal with child stations. */

    return 0;
}

#ifdef __cplusplus
}
#endif
