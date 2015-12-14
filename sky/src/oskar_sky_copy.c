/*
 * Copyright (c) 2015, The University of Oxford
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

#include <oskar_sky.h>
#include <private_sky.h>
#include <oskar_mem.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_copy(oskar_Sky* dst, const oskar_Sky* src, int* status)
{
    int num_sources;

    /* Check if safe to proceed. */
    if (*status) return;

    if (oskar_sky_precision(dst) != oskar_sky_precision(src))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return;
    }
    if (oskar_sky_capacity(dst) < oskar_sky_capacity(src))
    {
        *status = OSKAR_ERR_DIMENSION_MISMATCH;
        return;
    }

    /* Copy meta data */
    num_sources = src->num_sources;
    dst->num_sources = num_sources;
    dst->use_extended = src->use_extended;
    dst->reference_ra_rad = src->reference_ra_rad;
    dst->reference_dec_rad = src->reference_dec_rad;

    /* Copy the memory blocks */
    oskar_mem_copy_contents(dst->ra_rad, src->ra_rad,
            0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->dec_rad, src->dec_rad,
            0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->I, src->I, 0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->Q, src->Q, 0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->U, src->U, 0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->V, src->V, 0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->reference_freq_hz, src->reference_freq_hz,
            0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->spectral_index, src->spectral_index,
            0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->rm_rad, src->rm_rad,
            0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->l, src->l, 0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->m, src->m, 0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->n, src->n, 0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->fwhm_major_rad, src->fwhm_major_rad,
            0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->fwhm_minor_rad, src->fwhm_minor_rad,
            0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->pa_rad, src->pa_rad,
            0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->gaussian_a, src->gaussian_a,
            0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->gaussian_b, src->gaussian_b,
            0, 0, num_sources, status);
    oskar_mem_copy_contents(dst->gaussian_c, src->gaussian_c,
            0, 0, num_sources, status);
}

#ifdef __cplusplus
}
#endif
