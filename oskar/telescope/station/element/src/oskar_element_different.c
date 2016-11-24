/*
 * Copyright (c) 2015-2016, The University of Oxford
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

#include "telescope/station/element/private_element.h"
#include "telescope/station/element/oskar_element.h"

#ifdef __cplusplus
extern "C" {
#endif

int oskar_element_different(const oskar_Element* a, const oskar_Element* b,
        int* status)
{
    int i;

    /* Check if safe to proceed. */
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
            return 1;
        if (oskar_mem_different(a->filename_y[i], b->filename_y[i], 0, status))
            return 1;
        if (oskar_mem_different(a->filename_scalar[i], b->filename_scalar[i],
                0, status))
            return 1;
    }

    /* Elements are the same. */
    return 0;
}

#ifdef __cplusplus
}
#endif
