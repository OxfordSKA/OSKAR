/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
    int capacity = 0;

    /* Check type and location. */
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }

    /* Allocate and initialise a sky model structure. */
    model = (oskar_Sky*) calloc(1, sizeof(oskar_Sky));
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
