/*
 * Copyright (c) 2011, The University of Oxford
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

#include "interferometry/oskar_telescope_model_copy.h"
#include "interferometry/oskar_TelescopeModel.h"
#include "station/oskar_station_model_copy.h"
#include "utility/oskar_mem_copy.h"
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

int oskar_telescope_model_copy(oskar_TelescopeModel* dst,
        const oskar_TelescopeModel* src)
{
    int error = OSKAR_SUCCESS, i = 0;

    /* Ensure there is enough room in the station array. */
    dst->station = realloc(dst->station,
            src->num_stations * sizeof(oskar_StationModel));

    /* Copy each station. */
    for (i = 0; i < src->num_stations; ++i)
    {
        error = oskar_station_model_copy(&(dst->station[i]),
                &(src->station[i]));
        if (error) return error;
    }

    /* Copy the coordinates. */
    error = oskar_mem_copy(&dst->station_x, &src->station_x);
    if (error) return error;
    error = oskar_mem_copy(&dst->station_y, &src->station_y);
    if (error) return error;
    error = oskar_mem_copy(&dst->station_z, &src->station_z);
    if (error) return error;

    /* Copy remaining meta-data. */
    dst->num_stations = src->num_stations;
    dst->max_station_size = src->max_station_size;
    dst->coord_units = src->coord_units;
    dst->identical_stations = src->identical_stations;
    dst->disable_e_jones = src->disable_e_jones;
    dst->use_common_sky = src->use_common_sky;
    dst->seed_time_variable_errors = src->seed_time_variable_errors;
    dst->ra0_rad = src->ra0_rad;
    dst->dec0_rad = src->dec0_rad;
    dst->wavelength_metres = src->wavelength_metres;
    dst->bandwidth_hz = src->bandwidth_hz;

    return OSKAR_SUCCESS;
}

#ifdef __cplusplus
}
#endif
