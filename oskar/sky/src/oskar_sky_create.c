/*
 * Copyright (c) 2013-2016, The University of Oxford
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

#include "sky/private_sky.h"
#include "sky/oskar_sky.h"

#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Sky* oskar_sky_create(int type, int location, int num_sources,
        int* status)
{
    oskar_Sky* model = 0;
    int capacity;

    /* Check type and location. */
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }

    /* Allocate and initialise a sky model structure. */
    model = (oskar_Sky*) malloc(sizeof(oskar_Sky));
    if (!model)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }

    /* Set meta-data */
    capacity = num_sources + 1;
    model->precision = type;
    model->mem_location = location;
    model->capacity = capacity;
    model->num_sources = num_sources;
    model->use_extended = OSKAR_FALSE;
    model->reference_ra_rad = 0.0;
    model->reference_dec_rad = 0.0;

    /* Initialise the memory. */
    model->ra_rad = oskar_mem_create(type, location, capacity, status);
    model->dec_rad = oskar_mem_create(type, location, capacity, status);
    model->I = oskar_mem_create(type, location, capacity, status);
    model->Q = oskar_mem_create(type, location, capacity, status);
    model->U = oskar_mem_create(type, location, capacity, status);
    model->V = oskar_mem_create(type, location, capacity, status);
    model->reference_freq_hz = oskar_mem_create(type, location, capacity, status);
    model->spectral_index = oskar_mem_create(type, location, capacity, status);
    model->rm_rad = oskar_mem_create(type, location, capacity, status);
    model->l = oskar_mem_create(type, location, capacity, status);
    model->m = oskar_mem_create(type, location, capacity, status);
    model->n = oskar_mem_create(type, location, capacity, status);
    model->fwhm_major_rad = oskar_mem_create(type, location, capacity, status);
    model->fwhm_minor_rad = oskar_mem_create(type, location, capacity, status);
    model->pa_rad = oskar_mem_create(type, location, capacity, status);
    model->gaussian_a = oskar_mem_create(type, location, capacity, status);
    model->gaussian_b = oskar_mem_create(type, location, capacity, status);
    model->gaussian_c = oskar_mem_create(type, location, capacity, status);

    /* Return pointer to sky model. */
    return model;
}

#ifdef __cplusplus
}
#endif
