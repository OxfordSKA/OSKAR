/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "telescope/station/element/private_element.h"
#include "telescope/station/element/oskar_element.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_element_different(const oskar_Element* a, const oskar_Element* b,
        int* status)
{
    int i = 0;
    if (*status) return 1;

    if (a->precision != b->precision) return 1;

    if (a->coord_sys != b->coord_sys) return 1;
    if (a->max_radius_rad != b->max_radius_rad) return 1;
    if (a->x_element_type != b->x_element_type) return 1;
    if (a->y_element_type != b->y_element_type) return 1;
    if (a->x_taper_type != b->x_taper_type) return 1;
    if (a->y_taper_type != b->y_taper_type) return 1;
    if (a->x_dipole_length != b->x_dipole_length) return 1;
    if (a->y_dipole_length != b->y_dipole_length) return 1;
    if (a->x_dipole_length_units != b->x_dipole_length_units) return 1;
    if (a->y_dipole_length_units != b->y_dipole_length_units) return 1;
    if (a->x_taper_cosine_power != b->x_taper_cosine_power) return 1;
    if (a->y_taper_cosine_power != b->y_taper_cosine_power) return 1;
    if (a->x_taper_gaussian_fwhm_rad != b->x_taper_gaussian_fwhm_rad) return 1;
    if (a->y_taper_gaussian_fwhm_rad != b->y_taper_gaussian_fwhm_rad) return 1;
    if (a->x_taper_ref_freq_hz != b->x_taper_ref_freq_hz) return 1;
    if (a->y_taper_ref_freq_hz != b->y_taper_ref_freq_hz) return 1;

    /* Check frequency-dependent data. */
    if (a->num_freq != b->num_freq) return 1;
    for (i = 0; i < b->num_freq; ++i)
    {
        if (a->freqs_hz[i] != b->freqs_hz[i]) return 1;
        if (oskar_mem_different(a->filename_x[i], b->filename_x[i], 0, status))
        {
            return 1;
        }
        if (oskar_mem_different(a->filename_y[i], b->filename_y[i], 0, status))
        {
            return 1;
        }
        if (oskar_mem_different(a->filename_scalar[i], b->filename_scalar[i],
                0, status))
        {
            return 1;
        }
    }

    /* Elements are the same. */
    return 0;
}

#ifdef __cplusplus
}
#endif
