/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "sky/oskar_sky.h"
#include "sky/private_sky.h"
#include "mem/oskar_mem.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_sky_copy(oskar_Sky* dst, const oskar_Sky* src, int* status)
{
    int num_sources = 0;
    if (*status) return;

    if (oskar_sky_precision(dst) != oskar_sky_precision(src))
    {
        *status = OSKAR_ERR_TYPE_MISMATCH;
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
