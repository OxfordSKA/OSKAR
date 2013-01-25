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

#include "apps/lib/oskar_set_up_visibilities.h"
#include "interferometry/oskar_visibilities_init.h"
#include "utility/oskar_mem_type_check.h"
#include "utility/oskar_mem_copy.h"

#include <stdlib.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifdef __cplusplus
extern "C" {
#endif

void oskar_set_up_visibilities(oskar_Visibilities* vis,
        const oskar_Settings* settings, const oskar_TelescopeModel* telescope,
        int type, int* status)
{
    int num_stations, num_channels;

    /* Check all inputs. */
    if (!vis || !settings || !telescope || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    /* Check the type. */
    if (!oskar_mem_is_complex(type))
        *status = OSKAR_ERR_BAD_DATA_TYPE;

    /* Initialise the global visibility structure on the CPU. */
    num_stations = telescope->num_stations;
    num_channels = settings->obs.num_channels;
    oskar_visibilities_init(vis, type, OSKAR_LOCATION_CPU,
            num_channels, settings->obs.num_time_steps, num_stations, status);

    /* Add meta-data. */
    vis->freq_start_hz = settings->obs.start_frequency_hz;
    vis->freq_inc_hz = settings->obs.frequency_inc_hz;
    vis->time_start_mjd_utc = settings->obs.start_mjd_utc;
    vis->time_inc_seconds = settings->obs.dt_dump_days * 86400.0;
    vis->channel_bandwidth_hz = settings->obs.start_frequency_hz;
    vis->phase_centre_ra_deg = settings->obs.ra0_rad[0] * 180.0 / M_PI;
    vis->phase_centre_dec_deg = settings->obs.dec0_rad[0] * 180.0 / M_PI;

    /* Add settings file path. */
    oskar_mem_copy(&vis->settings_path, &settings->settings_path, status);

    /* Copy station coordinates from telescope model. */
    oskar_mem_copy(&vis->x_metres, &telescope->station_x, status);
    oskar_mem_copy(&vis->y_metres, &telescope->station_y, status);
    oskar_mem_copy(&vis->z_metres, &telescope->station_z, status);
}

#ifdef __cplusplus
}
#endif
