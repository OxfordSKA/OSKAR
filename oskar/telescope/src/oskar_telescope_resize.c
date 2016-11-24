/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include "telescope/private_telescope.h"
#include "telescope/oskar_telescope.h"

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_telescope_resize(oskar_Telescope* telescope, int size, int* status)
{
    int i, old_size = 0;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Get the old size. */
    old_size = telescope->num_stations;

    /* Check if increasing or decreasing in size. */
    if (size > old_size)
    {
        /* Enlarge the station array and create new stations. */
        telescope->station = realloc(telescope->station,
                size * sizeof(oskar_Station*));
        if (!telescope->station)
        {
            *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
            return;
        }
        for (i = old_size; i < size; ++i)
        {
            telescope->station[i] = oskar_station_create(
                    oskar_mem_type(telescope->station_true_x_offset_ecef_metres),
                    oskar_mem_location(telescope->station_true_x_offset_ecef_metres),
                    0, status);
        }
    }
    else if (size < old_size)
    {
        /* Free old stations and shrink the station array. */
        for (i = size; i < old_size; ++i)
        {
            oskar_station_free(oskar_telescope_station(telescope, i), status);
        }
        telescope->station = realloc(telescope->station,
                size * sizeof(oskar_Station*));
    }
    else
    {
        /* No resize necessary. */
        return;
    }

    /* Resize the remaining arrays. */
    oskar_mem_realloc(telescope->station_true_x_offset_ecef_metres,
            size, status);
    oskar_mem_realloc(telescope->station_true_y_offset_ecef_metres,
            size, status);
    oskar_mem_realloc(telescope->station_true_z_offset_ecef_metres,
            size, status);
    oskar_mem_realloc(telescope->station_true_x_enu_metres,
            size, status);
    oskar_mem_realloc(telescope->station_true_y_enu_metres,
            size, status);
    oskar_mem_realloc(telescope->station_true_z_enu_metres,
            size, status);
    oskar_mem_realloc(telescope->station_measured_x_offset_ecef_metres,
            size, status);
    oskar_mem_realloc(telescope->station_measured_y_offset_ecef_metres,
            size, status);
    oskar_mem_realloc(telescope->station_measured_z_offset_ecef_metres,
            size, status);
    oskar_mem_realloc(telescope->station_measured_x_enu_metres,
            size, status);
    oskar_mem_realloc(telescope->station_measured_y_enu_metres,
            size, status);
    oskar_mem_realloc(telescope->station_measured_z_enu_metres,
            size, status);

    /* Store the new size. */
    telescope->num_stations = size;
}

#ifdef __cplusplus
}
#endif
