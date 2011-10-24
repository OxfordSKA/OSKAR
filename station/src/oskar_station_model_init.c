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

#include "station/oskar_station_model_init.h"
#include "utility/oskar_mem_init.h"

int oskar_station_model_init(oskar_StationModel* model, int type, int location,
        int n_elements)
{
    // Check for sane inputs.
    if (model == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    // Check the type.
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
        return OSKAR_ERR_BAD_DATA_TYPE;

    // Determine the complex type.
    int complex_type = (type == OSKAR_SINGLE) ?
            OSKAR_SINGLE_COMPLEX : OSKAR_DOUBLE_COMPLEX;

    // Initialise variables.
    model->longitude = 0.0;
    model->latitude = 0.0;
    model->altitude = 0.0;
    model->ra0 = 0.0;
    model->dec0 = 0.0;
    model->n_elements = n_elements;

    // Initialise the memory.
    int error = 0;
    error = oskar_mem_init(&model->x, type, location, n_elements);
    if (error) return error;
    error = oskar_mem_init(&model->y, type, location, n_elements);
    if (error) return error;
    error = oskar_mem_init(&model->z, type, location, n_elements);
    if (error) return error;
    error = oskar_mem_init(&model->weight, complex_type, location, n_elements);
    if (error) return error;
    error = oskar_mem_init(&model->amp_gain, type, location, n_elements);
    if (error) return error;
    error = oskar_mem_init(&model->amp_error, type, location, n_elements);
    if (error) return error;
    error = oskar_mem_init(&model->phase_offset, type, location, n_elements);
    if (error) return error;
    error = oskar_mem_init(&model->phase_error, type, location, n_elements);
    if (error) return error;
    model->child = NULL;
    model->parent = NULL;
    model->single_element_model = 0;
    model->element_pattern = NULL;
    model->bit_depth = 0;
    return error;
}
