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
#include <oskar_sky.h>

#include <oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_append(oskar_Sky* dst, const oskar_Sky* src,
        int* status)
{
    /* Check all inputs. */
    if (!dst || !src || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Append to the sky model. */
    oskar_mem_append(dst->RA, src->RA, status);
    oskar_mem_append(dst->Dec, src->Dec, status);
    oskar_mem_append(dst->I, src->I, status);
    oskar_mem_append(dst->Q, src->Q, status);
    oskar_mem_append(dst->U, src->U, status);
    oskar_mem_append(dst->V, src->V, status);
    oskar_mem_append(dst->reference_freq, src->reference_freq, status);
    oskar_mem_append(dst->spectral_index, src->spectral_index, status);
    oskar_mem_append(dst->RM, src->RM, status);
    oskar_mem_append(dst->FWHM_major, src->FWHM_major, status);
    oskar_mem_append(dst->FWHM_minor, src->FWHM_minor, status);
    oskar_mem_append(dst->position_angle, src->position_angle, status);

    /* Update the number of sources. */
    dst->num_sources += src->num_sources;

    /* Resize arrays to hold the direction cosines. */
    oskar_mem_realloc(dst->l, dst->num_sources, status);
    oskar_mem_realloc(dst->m, dst->num_sources, status);
    oskar_mem_realloc(dst->n, dst->num_sources, status);

    /* Resize arrays to hold gaussian source parameters */
    oskar_mem_realloc(dst->gaussian_a, dst->num_sources, status);
    oskar_mem_realloc(dst->gaussian_b, dst->num_sources, status);
    oskar_mem_realloc(dst->gaussian_c, dst->num_sources, status);

    dst->use_extended = (src->use_extended || dst->use_extended);
}

#ifdef __cplusplus
}
#endif
