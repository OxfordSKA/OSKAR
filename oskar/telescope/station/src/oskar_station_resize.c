/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
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
    int feed = 0, dim = 0;
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
        int offset = 0, num_new = 0;
        offset = station->num_elements;
        num_new = num_elements - offset;

        /* Must set default element weight and gain. */
        for (feed = 0; feed < 2; feed++)
        {
            if (station->element_gain[feed])
            {
                oskar_mem_set_value_real(station->element_gain[feed],
                        1.0, offset, num_new, status);
            }
            if (station->element_weight[feed])
            {
                oskar_mem_set_value_real(station->element_weight[feed],
                        1.0, offset, num_new, status);
            }
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
