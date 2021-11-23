/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
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
        oskar_mem_copy(dst->filename_scalar[i], src->filename_scalar[i], status);
        if (src->x_v_re[i] && !dst->x_v_re[i])
        {
            dst->x_v_re[i] = oskar_splines_create(prec, loc, status);
            dst->x_v_im[i] = oskar_splines_create(prec, loc, status);
            dst->x_h_re[i] = oskar_splines_create(prec, loc, status);
            dst->x_h_im[i] = oskar_splines_create(prec, loc, status);
        }
        oskar_splines_copy(dst->x_v_re[i], src->x_v_re[i], status);
        oskar_splines_copy(dst->x_v_im[i], src->x_v_im[i], status);
        oskar_splines_copy(dst->x_h_re[i], src->x_h_re[i], status);
        oskar_splines_copy(dst->x_h_im[i], src->x_h_im[i], status);
        if (src->y_v_re[i] && !dst->y_v_re[i])
        {
            dst->y_v_re[i] = oskar_splines_create(prec, loc, status);
            dst->y_v_im[i] = oskar_splines_create(prec, loc, status);
            dst->y_h_re[i] = oskar_splines_create(prec, loc, status);
            dst->y_h_im[i] = oskar_splines_create(prec, loc, status);
        }
        oskar_splines_copy(dst->y_v_re[i], src->y_v_re[i], status);
        oskar_splines_copy(dst->y_v_im[i], src->y_v_im[i], status);
        oskar_splines_copy(dst->y_h_re[i], src->y_h_re[i], status);
        oskar_splines_copy(dst->y_h_im[i], src->y_h_im[i], status);
        if (src->scalar_re[i] && !dst->scalar_re[i])
        {
            dst->scalar_re[i] = oskar_splines_create(prec, loc, status);
            dst->scalar_im[i] = oskar_splines_create(prec, loc, status);
        }
        oskar_splines_copy(dst->scalar_re[i], src->scalar_re[i], status);
        oskar_splines_copy(dst->scalar_im[i], src->scalar_im[i], status);
        if (src->sph_wave[i] && !dst->sph_wave[i])
        {
            dst->sph_wave[i] = oskar_mem_create(sph_wave_type, loc, 0, status);
        }
        oskar_mem_copy(dst->sph_wave[i], src->sph_wave[i], status);
    }
}

#ifdef __cplusplus
}
#endif
