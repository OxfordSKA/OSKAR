/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include "station/oskar_station_model_resize.h"
#include "utility/oskar_mem_realloc.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_model_resize(oskar_StationModel* station, int n_elements,
        int* status)
{
    /* Check all inputs. */
    if (!station || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Set the new number of elements. */
    station->num_elements = n_elements;

    /* Resize the model data. */
    oskar_mem_realloc(&station->x_signal, n_elements, status);
    oskar_mem_realloc(&station->y_signal, n_elements, status);
    oskar_mem_realloc(&station->z_signal, n_elements, status);
    oskar_mem_realloc(&station->x_weights, n_elements, status);
    oskar_mem_realloc(&station->y_weights, n_elements, status);
    oskar_mem_realloc(&station->z_weights, n_elements, status);
    oskar_mem_realloc(&station->weight, n_elements, status);
    oskar_mem_realloc(&station->gain, n_elements, status);
    oskar_mem_realloc(&station->gain_error, n_elements, status);
    oskar_mem_realloc(&station->phase_offset, n_elements, status);
    oskar_mem_realloc(&station->phase_error, n_elements, status);
    oskar_mem_realloc(&station->cos_orientation_x, n_elements, status);
    oskar_mem_realloc(&station->sin_orientation_x, n_elements, status);
    oskar_mem_realloc(&station->cos_orientation_y, n_elements, status);
    oskar_mem_realloc(&station->sin_orientation_y, n_elements, status);
    oskar_mem_realloc(&station->element_type, n_elements, status);
}

#ifdef __cplusplus
}
#endif
