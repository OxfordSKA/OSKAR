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


#include "sky/oskar_sky_model_get_ptr.h"
#include "utility/oskar_mem_get_pointer.h"
#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_sky_model_get_ptr(oskar_SkyModel* sky_ptr, const oskar_SkyModel* sky,
        int offset, int num_sources)
{
    int err = OSKAR_SUCCESS;

    if (sky == NULL || sky_ptr == NULL)
        return OSKAR_ERR_INVALID_ARGUMENT;

    if (offset < 0 || num_sources < 0 || sky->num_sources < offset + num_sources)
        return OSKAR_ERR_OUT_OF_RANGE;

    sky_ptr->num_sources  = num_sources;
    sky_ptr->use_extended = sky->use_extended;
    err = oskar_mem_get_pointer(&sky_ptr->RA,  &sky->RA, offset, num_sources);
    if (err) return err;
    err = oskar_mem_get_pointer(&sky_ptr->Dec, &sky->Dec, offset, num_sources);
    if (err) return err;
    err = oskar_mem_get_pointer(&sky_ptr->I, &sky->I, offset, num_sources);
    if (err) return err;
    err = oskar_mem_get_pointer(&sky_ptr->Q, &sky->Q, offset, num_sources);
    if (err) return err;
    err = oskar_mem_get_pointer(&sky_ptr->U, &sky->U, offset, num_sources);
    if (err) return err;
    err = oskar_mem_get_pointer(&sky_ptr->V, &sky->V, offset, num_sources);
    if (err) return err;
    err = oskar_mem_get_pointer(&sky_ptr->reference_freq, &sky->reference_freq,
            offset, num_sources);
    if (err) return err;
    err = oskar_mem_get_pointer(&sky_ptr->spectral_index, &sky->spectral_index,
            offset, num_sources);
    if (err) return err;
    err = oskar_mem_get_pointer(&sky_ptr->rel_l, &sky->rel_l, offset, num_sources);
    if (err) return err;
    err = oskar_mem_get_pointer(&sky_ptr->rel_m, &sky->rel_m, offset, num_sources);
    if (err) return err;
    err = oskar_mem_get_pointer(&sky_ptr->rel_n, &sky->rel_n, offset, num_sources);
    if (err) return err;
    err = oskar_mem_get_pointer(&sky_ptr->FWHM_major, &sky->FWHM_major,
            offset, num_sources);
    if (err) return err;
    err = oskar_mem_get_pointer(&sky_ptr->FWHM_minor, &sky->FWHM_minor,
            offset, num_sources);
    if (err) return err;
    err = oskar_mem_get_pointer(&sky_ptr->position_angle, &sky->position_angle,
            offset, num_sources);
    if (err) return err;
    err = oskar_mem_get_pointer(&sky_ptr->gaussian_a, &sky->gaussian_a,
            offset, num_sources);
    if (err) return err;
    err = oskar_mem_get_pointer(&sky_ptr->gaussian_b, &sky->gaussian_b,
            offset, num_sources);
    if (err) return err;
    err = oskar_mem_get_pointer(&sky_ptr->gaussian_c, &sky->gaussian_c,
            offset, num_sources);
    if (err) return err;

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
