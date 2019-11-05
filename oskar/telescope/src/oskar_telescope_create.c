/*
 * Copyright (c) 2013-2019, The University of Oxford
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

#include "mem/oskar_mem.h"

#include <stdlib.h>
#include <float.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Telescope* oskar_telescope_create(int type, int location,
        int num_stations, int* status)
{
    int i = 0;
    oskar_Telescope* telescope;
    if (type != OSKAR_SINGLE && type != OSKAR_DOUBLE)
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }
    telescope = (oskar_Telescope*) calloc(1, sizeof(oskar_Telescope));
    if (!telescope)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }
    telescope->precision = type;
    telescope->mem_location = location;
    telescope->pol_mode = OSKAR_POL_MODE_FULL;
    telescope->num_stations = num_stations;
    telescope->max_station_depth = 1;
    telescope->enable_numerical_patterns = 1;
    telescope->uv_filter_max = FLT_MAX;
    telescope->uv_filter_units = OSKAR_METRES;
    telescope->noise_seed = 1;
    telescope->station_true_x_offset_ecef_metres =
            oskar_mem_create(type, location, num_stations, status);
    telescope->station_true_y_offset_ecef_metres =
            oskar_mem_create(type, location, num_stations, status);
    telescope->station_true_z_offset_ecef_metres =
            oskar_mem_create(type, location, num_stations, status);
    telescope->station_true_x_enu_metres =
            oskar_mem_create(type, location, num_stations, status);
    telescope->station_true_y_enu_metres =
            oskar_mem_create(type, location, num_stations, status);
    telescope->station_true_z_enu_metres =
            oskar_mem_create(type, location, num_stations, status);
    telescope->station_measured_x_offset_ecef_metres =
            oskar_mem_create(type, location, num_stations, status);
    telescope->station_measured_y_offset_ecef_metres =
            oskar_mem_create(type, location, num_stations, status);
    telescope->station_measured_z_offset_ecef_metres =
            oskar_mem_create(type, location, num_stations, status);
    telescope->station_measured_x_enu_metres =
            oskar_mem_create(type, location, num_stations, status);
    telescope->station_measured_y_enu_metres =
            oskar_mem_create(type, location, num_stations, status);
    telescope->station_measured_z_enu_metres =
            oskar_mem_create(type, location, num_stations, status);
    telescope->tec_screen_path =
            oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
if (num_stations > 0)
        telescope->station = (oskar_Station**) calloc(
                num_stations, sizeof(oskar_Station*));
    for (i = 0; i < num_stations; ++i)
    {
        telescope->station[i] = oskar_station_create(type, location, 0, status);
    }
    return telescope;
}

#ifdef __cplusplus
}
#endif
