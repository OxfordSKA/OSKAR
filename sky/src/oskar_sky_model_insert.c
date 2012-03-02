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


#include "sky/oskar_sky_model_insert.h"
#include "utility/oskar_mem_insert.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_sky_model_insert(oskar_SkyModel* dst, const oskar_SkyModel* src,
        int offset)
{
    int error = OSKAR_SUCCESS;

    error = oskar_mem_insert(&dst->RA, &src->RA, offset);
    if (error) return error;
    error = oskar_mem_insert(&dst->Dec, &src->Dec, offset);
    if (error) return error;
    error = oskar_mem_insert(&dst->I, &src->I, offset);
    if (error) return error;
    error = oskar_mem_insert(&dst->Q, &src->Q, offset);
    if (error) return error;
    error = oskar_mem_insert(&dst->U, &src->U, offset);
    if (error) return error;
    error = oskar_mem_insert(&dst->V, &src->V, offset);
    if (error) return error;
    error = oskar_mem_insert(&dst->reference_freq, &src->reference_freq, offset);
    if (error) return error;
    error = oskar_mem_insert(&dst->spectral_index, &src->spectral_index, offset);
    if (error) return error;
    error = oskar_mem_insert(&dst->rel_l, &src->rel_l, offset);
    if (error) return error;
    error = oskar_mem_insert(&dst->rel_m, &src->rel_m, offset);
    if (error) return error;
    error = oskar_mem_insert(&dst->rel_n, &src->rel_n, offset);
    if (error) return error;
    error = oskar_mem_insert(&dst->FWHM_major, &src->FWHM_major, offset);
    if (error) return error;
    error = oskar_mem_insert(&dst->FWHM_minor, &src->FWHM_minor, offset);
    if (error) return error;
    error = oskar_mem_insert(&dst->position_angle, &src->position_angle, offset);
    if (error) return error;
    error = oskar_mem_insert(&dst->gaussian_a, &src->gaussian_a, offset);
    if (error) return error;
    error = oskar_mem_insert(&dst->gaussian_b, &src->gaussian_b, offset);
    if (error) return error;
    error = oskar_mem_insert(&dst->gaussian_c, &src->gaussian_c, offset);
    if (error) return error;

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
