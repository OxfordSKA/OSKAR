/*
 * Copyright (c) 2012-2025, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/element/private_element.h"
#include "telescope/station/element/oskar_element.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_element_copy(oskar_Element* dst, const oskar_Element* src,
        int* status)
{
    int i = 0;
    if (*status) return;
    dst->precision = src->precision;
    dst->element_type = src->element_type;
    dst->taper_type = src->taper_type;
    dst->cosine_power = src->cosine_power;
    dst->gaussian_fwhm_rad = src->gaussian_fwhm_rad;
    dst->dipole_length = src->dipole_length;
    dst->dipole_length_units = src->dipole_length_units;
    oskar_element_resize_freq_data(dst, src->num_freq, status);
    const int prec = dst->precision;
    const int loc = dst->mem_location;
    const int sph_wave_type = prec | OSKAR_COMPLEX | OSKAR_MATRIX;
    for (i = 0; i < src->num_freq; ++i)
    {
        dst->freqs_hz[i] = src->freqs_hz[i];
        dst->l_max[i] = src->l_max[i];
        dst->common_phi_coords[i] = src->common_phi_coords[i];
        oskar_mem_copy(dst->filename_x[i], src->filename_x[i], status);
        oskar_mem_copy(dst->filename_y[i], src->filename_y[i], status);
        if (src->sph_wave[i] && !dst->sph_wave[i])
        {
            dst->sph_wave[i] = oskar_mem_create(sph_wave_type, loc, 0, status);
        }
        oskar_mem_copy(dst->sph_wave[i], src->sph_wave[i], status);
        if (src->sph_wave_feko[i] && !dst->sph_wave_feko[i])
        {
            dst->sph_wave_feko[i] = oskar_mem_create(
                    sph_wave_type, loc, 0, status
            );
        }
        oskar_mem_copy(dst->sph_wave_feko[i], src->sph_wave_feko[i], status);
        if (src->sph_wave_galileo[i] && !dst->sph_wave_galileo[i])
        {
            dst->sph_wave_galileo[i] = oskar_mem_create(
                    sph_wave_type, loc, 0, status
            );
        }
        oskar_mem_copy(
                dst->sph_wave_galileo[i], src->sph_wave_galileo[i], status
        );
    }
}

#ifdef __cplusplus
}
#endif
