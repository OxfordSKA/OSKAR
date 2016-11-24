/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#include "telescope/station/private_station.h"
#include "telescope/station/oskar_station.h"

#include "mem/oskar_mem.h"
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_station_different(const oskar_Station* a, const oskar_Station* b,
        int* status)
{
    int i, j, n, num_element_types;

    /* Check if safe to proceed. */
    if (*status) return 1;

    /* Don't check the unique ID, for obvious reasons! */
    /* Check if the meta-data are different. */
    n = a->num_elements;
    if (a->station_type != b->station_type ||
            a->normalise_final_beam != b->normalise_final_beam ||
            a->beam_coord_type != b->beam_coord_type ||
            a->beam_lon_rad != b->beam_lon_rad ||
            a->beam_lat_rad != b->beam_lat_rad ||
            a->pm_x_rad != b->pm_x_rad ||
            a->pm_y_rad != b->pm_y_rad ||
            a->identical_children != b->identical_children ||
            a->num_elements != b->num_elements ||
            a->num_element_types != b->num_element_types ||
            a->normalise_array_pattern != b->normalise_array_pattern ||
            a->enable_array_pattern != b->enable_array_pattern ||
            a->common_element_orientation != b->common_element_orientation ||
            a->array_is_3d != b->array_is_3d ||
            a->apply_element_errors != b->apply_element_errors ||
            a->apply_element_weight != b->apply_element_weight ||
            a->gaussian_beam_fwhm_rad != b->gaussian_beam_fwhm_rad ||
            a->gaussian_beam_reference_freq_hz != b->gaussian_beam_reference_freq_hz ||
            a->num_permitted_beams != b->num_permitted_beams)
    {
        return 1;
    }

    /* Check if child stations exist. */
    if ((oskar_station_has_child(a) && !oskar_station_has_child(b)) ||
            (!oskar_station_has_child(a) && oskar_station_has_child(b)))
        return 1;

    /* Check if element patterns exist. */
    if ( (oskar_station_has_element(a) && !oskar_station_has_element(b)) ||
            (!oskar_station_has_element(a) && oskar_station_has_element(b)) )
        return 1;

    /* Check if element pattern filenames are different,
     * for each element type. */
    num_element_types = oskar_station_num_element_types(a);
    for (j = 0; j < num_element_types; ++j)
    {
        const oskar_Element *e_a, *e_b;
        int num_freq_a, num_freq_b;
        e_a = oskar_station_element_const(a, j);
        e_b = oskar_station_element_const(b, j);

        /* Check if number of frequencies in element models are different. */
        num_freq_a = oskar_element_num_freq(e_a);
        num_freq_b = oskar_element_num_freq(e_b);
        if (num_freq_a != num_freq_b)
            return 1;

        for (i = 0; i < num_freq_a; ++i)
        {
            const oskar_Mem *fname_a_x = 0, *fname_a_y = 0;
            const oskar_Mem *fname_b_x = 0, *fname_b_y = 0;

            fname_a_x = oskar_element_x_filename_const(e_a, i);
            fname_a_y = oskar_element_y_filename_const(e_a, i);
            fname_b_x = oskar_element_x_filename_const(e_b, i);
            fname_b_y = oskar_element_y_filename_const(e_b, i);
            if (oskar_mem_different(fname_a_x, fname_b_x, 0, status))
                return 1;
            if (oskar_mem_different(fname_a_y, fname_b_y, 0, status))
                return 1;
        }
    }

    /* Check if the memory contents are different. */
    if (oskar_mem_different(a->noise_freq_hz, b->noise_freq_hz, 0, status))
        return 1;
    if (oskar_mem_different(a->noise_rms_jy, b->noise_rms_jy, 0, status))
        return 1;
    if (oskar_mem_different(a->element_measured_x_enu_metres,
            b->element_measured_x_enu_metres, n, status))
        return 1;
    if (oskar_mem_different(a->element_measured_y_enu_metres,
            b->element_measured_y_enu_metres, n, status))
        return 1;
    if (oskar_mem_different(a->element_measured_z_enu_metres,
            b->element_measured_z_enu_metres, n, status))
        return 1;
    if (oskar_mem_different(a->element_true_x_enu_metres,
            b->element_true_x_enu_metres, n, status))
        return 1;
    if (oskar_mem_different(a->element_true_y_enu_metres,
            b->element_true_y_enu_metres, n, status))
        return 1;
    if (oskar_mem_different(a->element_true_z_enu_metres,
            b->element_true_z_enu_metres, n, status))
        return 1;
    if (oskar_mem_different(a->element_gain, b->element_gain, n, status))
        return 1;
    if (oskar_mem_different(a->element_phase_offset_rad,
            b->element_phase_offset_rad, n, status))
        return 1;
    if (oskar_mem_different(a->element_weight, b->element_weight, n, status))
        return 1;
    if (oskar_mem_different(a->element_x_alpha_cpu,
            b->element_x_alpha_cpu, n, status))
        return 1;
    if (oskar_mem_different(a->element_x_beta_cpu,
            b->element_x_beta_cpu, n, status))
        return 1;
    if (oskar_mem_different(a->element_x_gamma_cpu,
            b->element_x_gamma_cpu, n, status))
        return 1;
    if (oskar_mem_different(a->element_y_alpha_cpu,
            b->element_y_alpha_cpu, n, status))
        return 1;
    if (oskar_mem_different(a->element_y_beta_cpu,
            b->element_y_beta_cpu, n, status))
        return 1;
    if (oskar_mem_different(a->element_y_gamma_cpu,
            b->element_y_gamma_cpu, n, status))
        return 1;
    if (oskar_mem_different(a->element_types, b->element_types, n, status))
        return 1;
    if (oskar_mem_different(a->element_types_cpu, b->element_types_cpu, n,
            status))
        return 1;
    if (oskar_mem_different(a->element_mount_types_cpu,
            b->element_mount_types_cpu, n, status))
        return 1;
    if (oskar_mem_different(a->permitted_beam_az_rad,
            b->permitted_beam_az_rad, n, status))
        return 1;
    if (oskar_mem_different(a->permitted_beam_el_rad,
            b->permitted_beam_el_rad, n, status))
        return 1;

    /* Recursively check child stations. */
    if (oskar_station_has_child(a) && oskar_station_has_child(b))
    {
        for (i = 0; i < n; ++i)
        {
            if (oskar_station_different(oskar_station_child_const(a, i),
                    oskar_station_child_const(b, i), status))
                return 1;
        }
    }

    /* Stations must be the same! */
    return 0;
}

#ifdef __cplusplus
}
#endif
