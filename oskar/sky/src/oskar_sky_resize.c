/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/private_sky.h"
#include "sky/oskar_sky.h"

#include "mem/oskar_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_resize(oskar_Sky* sky, int num_sources, int* status)
{
    if (*status) return;
    const int capacity = num_sources + 1;
    sky->capacity = capacity;
    sky->num_sources = num_sources;
    oskar_mem_realloc(sky->ra_rad, capacity, status);
    oskar_mem_realloc(sky->dec_rad, capacity, status);
    oskar_mem_realloc(sky->I, capacity, status);
    oskar_mem_realloc(sky->Q, capacity, status);
    oskar_mem_realloc(sky->U, capacity, status);
    oskar_mem_realloc(sky->V, capacity, status);
    oskar_mem_realloc(sky->reference_freq_hz, capacity, status);
    oskar_mem_realloc(sky->spectral_index, capacity, status);
    oskar_mem_realloc(sky->rm_rad, capacity, status);
    oskar_mem_realloc(sky->l, capacity, status);
    oskar_mem_realloc(sky->m, capacity, status);
    oskar_mem_realloc(sky->n, capacity, status);
    oskar_mem_realloc(sky->fwhm_major_rad, capacity, status);
    oskar_mem_realloc(sky->fwhm_minor_rad, capacity, status);
    oskar_mem_realloc(sky->pa_rad, capacity, status);
    oskar_mem_realloc(sky->gaussian_a, capacity, status);
    oskar_mem_realloc(sky->gaussian_b, capacity, status);
    oskar_mem_realloc(sky->gaussian_c, capacity, status);
}

#ifdef __cplusplus
}
#endif
