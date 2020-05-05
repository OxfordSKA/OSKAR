/*
 * Copyright (c) 2011-2020, The University of Oxford
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
#include "math/oskar_cmath.h"
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_station_resize(oskar_Station* station, int num_elements,
        int* status)
{
    int feed, dim;
    if (*status || !station) return;
    for (feed = 0; feed < 2; feed++)
    {
        for (dim = 0; dim < 3; dim++)
        {
            oskar_mem_realloc(station->element_true_enu_metres[feed][dim],
                    num_elements, status);
            oskar_mem_realloc(station->element_measured_enu_metres[feed][dim],
                    num_elements, status);
            oskar_mem_realloc(station->element_euler_cpu[feed][dim],
                    num_elements, status);
        }
        oskar_mem_realloc(station->element_weight[feed],
                num_elements, status);
        oskar_mem_realloc(station->element_cable_length_error[feed],
                num_elements, status);
        oskar_mem_realloc(station->element_gain[feed],
                num_elements, status);
        oskar_mem_realloc(station->element_gain_error[feed],
                num_elements, status);
        oskar_mem_realloc(station->element_phase_offset_rad[feed],
                num_elements, status);
        oskar_mem_realloc(station->element_phase_error_rad[feed],
                num_elements, status);
    }
    oskar_mem_realloc(station->element_types, num_elements, status);
    oskar_mem_realloc(station->element_types_cpu, num_elements, status);
    oskar_mem_realloc(station->element_mount_types_cpu, num_elements, status);

    /* Initialise any new elements with default values. */
    if (num_elements > station->num_elements)
    {
        int offset, num_new;
        offset = station->num_elements;
        num_new = num_elements - offset;

        /* Must set default element weight and gain. */
        for (feed = 0; feed < 2; feed++)
        {
            if (station->element_gain[feed])
                oskar_mem_set_value_real(station->element_gain[feed],
                        1.0, offset, num_new, status);
            if (station->element_weight[feed])
                oskar_mem_set_value_real(station->element_weight[feed],
                        1.0, offset, num_new, status);
        }
        memset(oskar_mem_char(station->element_mount_types_cpu) + offset,
                'F', num_new);
    }

    /* Set the new number of elements. */
    station->num_elements = num_elements;
}

#ifdef __cplusplus
}
#endif
