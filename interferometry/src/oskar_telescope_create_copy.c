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
    telescope->mem_location = location;

    /* Copy the meta-data. */
    telescope->num_stations = src->num_stations;
    telescope->max_station_size = src->max_station_size;
    telescope->max_station_depth = src->max_station_depth;
    telescope->identical_stations = src->identical_stations;
    telescope->use_common_sky = src->use_common_sky;
    telescope->seed_time_variable_station_element_errors =
            src->seed_time_variable_station_element_errors;
    telescope->lon_rad = src->lon_rad;
    telescope->lat_rad = src->lat_rad;
    telescope->alt_metres = src->alt_metres;
    telescope->phase_centre_ra_rad = src->phase_centre_ra_rad;
    telescope->phase_centre_dec_rad = src->phase_centre_dec_rad;
    telescope->channel_bandwidth_hz = src->channel_bandwidth_hz;
    telescope->time_average_sec = src->time_average_sec;
    telescope->uv_filter_min = src->uv_filter_min;
    telescope->uv_filter_max = src->uv_filter_max;
    telescope->uv_filter_units = src->uv_filter_units;

    /* Copy the coordinates. */
    oskar_mem_copy(telescope->station_true_x_offset_ecef_metres,
            src->station_true_x_offset_ecef_metres, status);
    oskar_mem_copy(telescope->station_true_y_offset_ecef_metres,
            src->station_true_y_offset_ecef_metres, status);
    oskar_mem_copy(telescope->station_true_z_offset_ecef_metres,
            src->station_true_z_offset_ecef_metres, status);
    oskar_mem_copy(telescope->station_true_x_enu_metres,
            src->station_true_x_enu_metres, status);
    oskar_mem_copy(telescope->station_true_y_enu_metres,
            src->station_true_y_enu_metres, status);
    oskar_mem_copy(telescope->station_true_z_enu_metres,
            src->station_true_z_enu_metres, status);

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
