/*
 * Copyright (c) 2013-2014, The University of Oxford
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

#include <private_station.h>
#include <oskar_station.h>

#include <oskar_mem.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_station_different(const oskar_Station* a, const oskar_Station* b,
        int* status)
{
    int i, n;
    const oskar_Mem *fname_a_x = 0, *fname_a_y = 0;
    const oskar_Mem *fname_b_x = 0, *fname_b_y = 0;

    /* Check all inputs. */
    if (!a || !b || !status)
    {
        oskar_set_invalid_argument(status);
        return 1;
    }

    /* Check if safe to proceed. */
    if (*status) return 1;

    /* Check if the meta-data are different. */
    n = a->num_elements;
    if (a->station_type != b->station_type ||
            a->beam_coord_type != b->beam_coord_type ||
            a->beam_longitude_rad != b->beam_longitude_rad ||
            a->beam_latitude_rad != b->beam_latitude_rad ||
            a->identical_children != b->identical_children ||
            a->num_elements != b->num_elements ||
            a->num_element_types != b->num_element_types ||
            a->use_polarised_elements != b->use_polarised_elements ||
            a->normalise_beam != b->normalise_beam ||
            a->enable_array_pattern != b->enable_array_pattern ||
            a->single_element_model != b->single_element_model ||
            a->array_is_3d != b->array_is_3d ||
            a->apply_element_errors != b->apply_element_errors ||
            a->apply_element_weight != b->apply_element_weight ||
            a->gaussian_beam_fwhm_rad != b->gaussian_beam_fwhm_rad ||
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

    /* FIXME Check if element pattern filenames are different (needs updating for multiple element types). */
    if (oskar_station_has_element(a))
    {
        const oskar_Element* e;
        e = oskar_station_element_const(a, 0);
        fname_a_x = oskar_element_x_filename_const(e);
        fname_a_y = oskar_element_y_filename_const(e);
    }
    if (oskar_station_has_element(b))
    {
        const oskar_Element* e;
        e = oskar_station_element_const(b, 0);
        fname_b_x = oskar_element_x_filename_const(e);
        fname_b_y = oskar_element_y_filename_const(e);
    }
    if (oskar_mem_different(fname_a_x, fname_b_x, 0, status))
        return 1;
    if (oskar_mem_different(fname_a_y, fname_b_y, 0, status))
        return 1;

    /* Check if the memory contents are different. */
    if (oskar_mem_different(a->x_weights, b->x_weights, n, status))
        return 1;
    if (oskar_mem_different(a->y_weights, b->y_weights, n, status))
        return 1;
    if (oskar_mem_different(a->z_weights, b->z_weights, n, status))
        return 1;
    if (oskar_mem_different(a->x_signal, b->x_signal, n, status))
        return 1;
    if (oskar_mem_different(a->y_signal, b->y_signal, n, status))
        return 1;
    if (oskar_mem_different(a->z_signal, b->z_signal, n, status))
        return 1;
    if (oskar_mem_different(a->gain, b->gain, n, status))
        return 1;
    if (oskar_mem_different(a->phase_offset, b->phase_offset, n, status))
        return 1;
    if (oskar_mem_different(a->weight, b->weight, n, status))
        return 1;
    if (oskar_mem_different(a->cos_orientation_x, b->cos_orientation_x, n,
            status))
        return 1;
    if (oskar_mem_different(a->sin_orientation_x, b->sin_orientation_x, n,
            status))
        return 1;
    if (oskar_mem_different(a->cos_orientation_y, b->cos_orientation_y, n,
            status))
        return 1;
    if (oskar_mem_different(a->sin_orientation_y, b->sin_orientation_y, n,
            status))
        return 1;
    if (oskar_mem_different(a->element_type, b->element_type, n, status))
        return 1;
    if (oskar_mem_different(a->permitted_beam_az, b->permitted_beam_az, n,
            status))
        return 1;
    if (oskar_mem_different(a->permitted_beam_el, b->permitted_beam_el, n,
            status))
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
