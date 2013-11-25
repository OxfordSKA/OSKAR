/*
 * Copyright (c) 2013, The University of Oxford
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

#include <private_telescope.h>
#include <oskar_telescope.h>

#include <stdlib.h>
#include <stdio.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Telescope* oskar_telescope_create_copy(const oskar_Telescope* src,
        int location, int* status)
{
    int i = 0;
    oskar_Telescope* telescope;

    /* Check all inputs. */
    if (!status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Create a new, empty model. */
    telescope = oskar_telescope_create(oskar_telescope_precision(src),
            location, 0, status);

    /* Copy private meta-data. */
    telescope->precision = src->precision;
    telescope->location = location;

    /* Copy the meta-data. */
    telescope->num_stations = src->num_stations;
    telescope->max_station_size = src->max_station_size;
    telescope->max_station_depth = src->max_station_depth;
    telescope->identical_stations = src->identical_stations;
    telescope->use_common_sky = src->use_common_sky;
    telescope->seed_time_variable_station_element_errors =
            src->seed_time_variable_station_element_errors;
    telescope->longitude_rad = src->longitude_rad;
    telescope->latitude_rad = src->latitude_rad;
    telescope->altitude_m = src->altitude_m;
    telescope->ra0_rad = src->ra0_rad;
    telescope->dec0_rad = src->dec0_rad;
    telescope->bandwidth_hz = src->bandwidth_hz;
    telescope->time_average_sec = src->time_average_sec;

    /* Copy the coordinates. */
    oskar_mem_copy(&telescope->station_x, &src->station_x, status);
    oskar_mem_copy(&telescope->station_y, &src->station_y, status);
    oskar_mem_copy(&telescope->station_z, &src->station_z, status);
    oskar_mem_copy(&telescope->station_x_hor, &src->station_x_hor, status);
    oskar_mem_copy(&telescope->station_y_hor, &src->station_y_hor, status);
    oskar_mem_copy(&telescope->station_z_hor, &src->station_z_hor, status);

    /* Copy each station. */
    telescope->station = malloc(src->num_stations * sizeof(oskar_Station*));
    for (i = 0; i < src->num_stations; ++i)
    {
        telescope->station[i] = oskar_station_create_copy(
                oskar_telescope_station_const(src, i), location, status);
    }

    /* Return pointer to data structure. */
    return telescope;
}

#ifdef __cplusplus
}
#endif
