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
#include "mem/oskar_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

oskar_Sky* oskar_sky_create_copy(const oskar_Sky* src, int location,
        int* status)
{
    oskar_Sky* model = 0;

    /* Check if safe to proceed. */
    if (*status) return model;

    /* Create the new model. */
    model = oskar_sky_create(oskar_sky_precision(src), location,
            oskar_sky_num_sources(src), status);

    /* Copy meta data */
    model->precision = src->precision;
    model->mem_location = location;
    model->capacity = src->capacity;
    model->num_sources = src->num_sources;
    model->use_extended = src->use_extended;
    model->reference_ra_rad = src->reference_ra_rad;
    model->reference_dec_rad = src->reference_dec_rad;

    /* Copy the memory blocks */
    oskar_mem_copy(model->ra_rad, src->ra_rad, status);
    oskar_mem_copy(model->dec_rad, src->dec_rad, status);
    oskar_mem_copy(model->I, src->I, status);
    oskar_mem_copy(model->Q, src->Q, status);
    oskar_mem_copy(model->U, src->U, status);
    oskar_mem_copy(model->V, src->V, status);
    oskar_mem_copy(model->reference_freq_hz, src->reference_freq_hz, status);
    oskar_mem_copy(model->spectral_index, src->spectral_index, status);
    oskar_mem_copy(model->rm_rad, src->rm_rad, status);
    oskar_mem_copy(model->l, src->l, status);
    oskar_mem_copy(model->m, src->m, status);
    oskar_mem_copy(model->n, src->n, status);
    oskar_mem_copy(model->fwhm_major_rad, src->fwhm_major_rad, status);
    oskar_mem_copy(model->fwhm_minor_rad, src->fwhm_minor_rad, status);
    oskar_mem_copy(model->pa_rad, src->pa_rad, status);
    oskar_mem_copy(model->gaussian_a, src->gaussian_a, status);
    oskar_mem_copy(model->gaussian_b, src->gaussian_b, status);
    oskar_mem_copy(model->gaussian_c, src->gaussian_c, status);

    /* Return pointer to new sky model. */
    return model;
}

#ifdef __cplusplus
}
#endif
