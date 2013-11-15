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

#include <private_station.h>
#include <oskar_station.h>

#include <oskar_element_copy.h>
#include <oskar_system_noise_model_copy.h>

#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Station* oskar_station_create_copy(const oskar_Station* src,
        int location, int* status)
{
    int i = 0;
    oskar_Station* model = 0;

    /* Check all inputs. */
    if (!src || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check if safe to proceed. */
    if (*status) return model;

    /* Create the new model. */
    model = oskar_station_create(oskar_station_precision(src), location,
            oskar_station_num_elements(src), status);

    /* Set meta-data. */
    model->precision = src->precision;
    model->location = location;

    /* Copy common station parameters. */
    model->station_type = src->station_type;
    model->longitude_rad = src->longitude_rad;
    model->latitude_rad = src->latitude_rad;
    model->altitude_m = src->altitude_m;
    model->beam_longitude_rad = src->beam_longitude_rad;
    model->beam_latitude_rad = src->beam_latitude_rad;
    model->beam_coord_type = src->beam_coord_type;
    oskar_system_noise_model_copy(&model->noise, &src->noise, status);

    /* Copy aperture array data, except num_element_types (done later). */
    model->num_elements = src->num_elements;
    model->use_polarised_elements = src->use_polarised_elements;
    model->normalise_beam = src->normalise_beam;
    model->enable_array_pattern = src->enable_array_pattern;
    model->single_element_model = src->single_element_model;
    model->array_is_3d = src->array_is_3d;
    model->apply_element_errors = src->apply_element_errors;
    model->apply_element_weight = src->apply_element_weight;
    model->orientation_x = src->orientation_x;
    model->orientation_y = src->orientation_y;

    /* Copy Gaussian station beam data. */
    model->gaussian_beam_fwhm_rad = src->gaussian_beam_fwhm_rad;

    /* Copy memory blocks. */
    oskar_mem_copy(&model->x_signal, &src->x_signal, status);
    oskar_mem_copy(&model->y_signal, &src->y_signal, status);
    oskar_mem_copy(&model->z_signal, &src->z_signal, status);
    oskar_mem_copy(&model->x_weights, &src->x_weights, status);
    oskar_mem_copy(&model->y_weights, &src->y_weights, status);
    oskar_mem_copy(&model->z_weights, &src->z_weights, status);
    oskar_mem_copy(&model->weight, &src->weight, status);
    oskar_mem_copy(&model->gain, &src->gain, status);
    oskar_mem_copy(&model->gain_error, &src->gain_error, status);
    oskar_mem_copy(&model->phase_offset, &src->phase_offset, status);
    oskar_mem_copy(&model->phase_error, &src->phase_error, status);
    oskar_mem_copy(&model->cos_orientation_x, &src->cos_orientation_x, status);
    oskar_mem_copy(&model->sin_orientation_x, &src->sin_orientation_x, status);
    oskar_mem_copy(&model->cos_orientation_y, &src->cos_orientation_y, status);
    oskar_mem_copy(&model->sin_orientation_y, &src->sin_orientation_y, status);
    oskar_mem_copy(&model->element_type, &src->element_type, status);

    /* Copy element models, if set. */
    if (oskar_station_has_element(src))
    {
        /* Ensure enough space for element model data. */
        oskar_station_resize_element_types(model, src->num_element_types,
                status);

        /* Copy the element model data. */
        for (i = 0; i < src->num_element_types; ++i)
        {
            oskar_element_copy(model->element_pattern[i],
                    src->element_pattern[i], status);
        }
    }

    /* Recursively copy child stations. */
    if (oskar_station_has_child(src))
    {
        model->child = malloc(src->num_elements * sizeof(oskar_Station*));
        if (!model->child)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            return model;
        }

        for (i = 0; i < src->num_elements; ++i)
        {
            model->child[i] = oskar_station_create_copy(
                    oskar_station_child_const(src, i), location, status);
        }
    }

    return model;
}

#ifdef __cplusplus
}
#endif
