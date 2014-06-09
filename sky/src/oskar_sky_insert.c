/*
 * Copyright (c) 2012-2014, The University of Oxford
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

void oskar_sky_insert(oskar_Sky* dst, const oskar_Sky* src, int offset,
        int* status)
{
    size_t num_sources;

    /* Check all inputs. */
    if (!src || !dst || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    num_sources = (size_t)oskar_sky_num_sources(src);
    oskar_mem_insert(dst->ra_rad, src->ra_rad, offset, num_sources, status);
    oskar_mem_insert(dst->dec_rad, src->dec_rad, offset, num_sources, status);
    oskar_mem_insert(dst->I, src->I, offset, num_sources, status);
    oskar_mem_insert(dst->Q, src->Q, offset, num_sources, status);
    oskar_mem_insert(dst->U, src->U, offset, num_sources, status);
    oskar_mem_insert(dst->V, src->V, offset, num_sources, status);
    oskar_mem_insert(dst->reference_freq_hz, src->reference_freq_hz, offset,
            num_sources, status);
    oskar_mem_insert(dst->spectral_index, src->spectral_index, offset,
            num_sources, status);
    oskar_mem_insert(dst->rm_rad, src->rm_rad, offset, num_sources, status);
    oskar_mem_insert(dst->l, src->l, offset, num_sources, status);
    oskar_mem_insert(dst->m, src->m, offset, num_sources, status);
    oskar_mem_insert(dst->n, src->n, offset, num_sources, status);
    oskar_mem_insert(dst->fwhm_major_rad, src->fwhm_major_rad, offset, num_sources,
            status);
    oskar_mem_insert(dst->fwhm_minor_rad, src->fwhm_minor_rad, offset, num_sources,
            status);
    oskar_mem_insert(dst->pa_rad, src->pa_rad, offset,
            num_sources, status);
    oskar_mem_insert(dst->gaussian_a, src->gaussian_a, offset, num_sources,
            status);
    oskar_mem_insert(dst->gaussian_b, src->gaussian_b, offset, num_sources,
            status);
    oskar_mem_insert(dst->gaussian_c, src->gaussian_c, offset, num_sources,
            status);
}

#ifdef __cplusplus
}
#endif
