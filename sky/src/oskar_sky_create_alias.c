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

#include <private_sky.h>
#include <oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Sky* oskar_sky_create_alias(const oskar_Sky* sky, int offset,
        int num_sources, int* status)
{
    oskar_Sky* sky_ptr = 0;

    /* Check all inputs. */
    if (!sky || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Check ranges. */
    if (offset < 0 || num_sources < 0 ||
            sky->num_sources < offset + num_sources)
    {
        *status = OSKAR_ERR_OUT_OF_RANGE;
        return 0;
    }

    /* Create the sky model structure. */
    sky_ptr = (oskar_Sky*) malloc(sizeof(oskar_Sky));

    /* Set meta-data */
    sky_ptr->precision = sky->precision;
    sky_ptr->mem_location = sky->mem_location;
    sky_ptr->num_sources = num_sources;
    sky_ptr->use_extended = sky->use_extended;
    sky_ptr->reference_ra_rad = sky->reference_ra_rad;
    sky_ptr->reference_dec_rad = sky->reference_dec_rad;

    sky_ptr->ra_rad = oskar_mem_create_alias(sky->ra_rad, offset, num_sources, status);
    sky_ptr->dec_rad = oskar_mem_create_alias(sky->dec_rad, offset, num_sources, status);
    sky_ptr->I = oskar_mem_create_alias(sky->I, offset, num_sources, status);
    sky_ptr->Q = oskar_mem_create_alias(sky->Q, offset, num_sources, status);
    sky_ptr->U = oskar_mem_create_alias(sky->U, offset, num_sources, status);
    sky_ptr->V = oskar_mem_create_alias(sky->V, offset, num_sources, status);
    sky_ptr->reference_freq_hz = oskar_mem_create_alias(sky->reference_freq_hz,
            offset, num_sources, status);
    sky_ptr->spectral_index = oskar_mem_create_alias(sky->spectral_index,
            offset, num_sources, status);
    sky_ptr->rm_rad = oskar_mem_create_alias(sky->rm_rad, offset, num_sources, status);
    sky_ptr->l = oskar_mem_create_alias(sky->l, offset, num_sources, status);
    sky_ptr->m = oskar_mem_create_alias(sky->m, offset, num_sources, status);
    sky_ptr->n = oskar_mem_create_alias(sky->n, offset, num_sources, status);
    sky_ptr->fwhm_major_rad = oskar_mem_create_alias(sky->fwhm_major_rad,
            offset, num_sources, status);
    sky_ptr->fwhm_minor_rad = oskar_mem_create_alias(sky->fwhm_minor_rad,
            offset, num_sources, status);
    sky_ptr->pa_rad = oskar_mem_create_alias(sky->pa_rad,
            offset, num_sources, status);
    sky_ptr->gaussian_a = oskar_mem_create_alias(sky->gaussian_a,
            offset, num_sources, status);
    sky_ptr->gaussian_b = oskar_mem_create_alias(sky->gaussian_b,
            offset, num_sources, status);
    sky_ptr->gaussian_c = oskar_mem_create_alias(sky->gaussian_c,
            offset, num_sources, status);

    return sky_ptr;
}

#ifdef __cplusplus
}
#endif
