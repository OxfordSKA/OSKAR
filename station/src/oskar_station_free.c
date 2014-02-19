/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#include <oskar_element_free.h>
#include <oskar_system_noise_model_free.h>

#include <oskar_mem.h>
#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_free(oskar_Station* model, int* status)
{
    int i = 0;

    /* Check all inputs. */
    if (!model || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Free the element data. */
    oskar_mem_free(model->x_signal, status);
    oskar_mem_free(model->y_signal, status);
    oskar_mem_free(model->z_signal, status);
    oskar_mem_free(model->x_weights, status);
    oskar_mem_free(model->y_weights, status);
    oskar_mem_free(model->z_weights, status);
    oskar_mem_free(model->weight, status);
    oskar_mem_free(model->gain, status);
    oskar_mem_free(model->gain_error, status);
    oskar_mem_free(model->phase_offset, status);
    oskar_mem_free(model->phase_error, status);
    oskar_mem_free(model->cos_orientation_x, status);
    oskar_mem_free(model->sin_orientation_x, status);
    oskar_mem_free(model->cos_orientation_y, status);
    oskar_mem_free(model->sin_orientation_y, status);
    oskar_mem_free(model->element_type, status);
    oskar_mem_free(model->permitted_beam_az, status);
    oskar_mem_free(model->permitted_beam_el, status);

    /* Free the noise model. */
    oskar_system_noise_model_free(model->noise, status);

    /* Free the element pattern data if it exists. */
    if (oskar_station_has_element(model))
    {
        for (i = 0; i < model->num_element_types; ++i)
        {
            oskar_element_free(oskar_station_element(model, i), status);
        }

        /* Free the element model handle array. */
        free(model->element_pattern);
    }

    /* Recursively free the child stations. */
    if (oskar_station_has_child(model))
    {
        for (i = 0; i < model->num_elements; ++i)
        {
            oskar_station_free(oskar_station_child(model, i), status);
        }

        /* Free the child station handle array. */
        free(model->child);
    }

    /* Free the structure itself. */
    free(model);
}

#ifdef __cplusplus
}
#endif
