/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include <private_sky.h>
#include <oskar_sky.h>

#include <oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_resize(oskar_Sky* sky, int num_sources, int* status)
{
    int capacity;

    /* Check if safe to proceed. */
    if (*status) return;

    capacity = num_sources + 1;
    sky->capacity = capacity;
    sky->num_sources = num_sources;

    /* Resize the model data. */
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
