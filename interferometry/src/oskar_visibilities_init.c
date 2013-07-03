/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include "interferometry/oskar_visibilities_init.h"
#include "interferometry/oskar_Visibilities.h"
#include "utility/oskar_mem_init.h"
#include "utility/oskar_mem_type_check.h"

#ifdef __cplusplus
extern "C" {
#endif

void oskar_visibilities_init(oskar_Visibilities* vis, int amp_type, int location,
        int num_channels, int num_times, int num_stations, int* status)
{
    int type, num_amps, num_coords, num_baselines;

    /* Check all inputs. */
    if (!vis || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    if (!oskar_mem_is_complex(amp_type))
        *status = OSKAR_ERR_BAD_DATA_TYPE;

    if (location != OSKAR_LOCATION_GPU && location != OSKAR_LOCATION_CPU)
        *status = OSKAR_ERR_BAD_LOCATION;

    /* Evaluate the coordinate type. */
    type = oskar_mem_is_double(amp_type) ? OSKAR_DOUBLE : OSKAR_SINGLE;

    /* Set dimensions. */
    num_baselines = num_stations * (num_stations - 1) / 2;
    vis->num_stations  = num_stations;
    vis->num_channels  = num_channels;
    vis->num_times     = num_times;
    vis->num_baselines = num_baselines;
    num_amps   = num_channels * num_times * num_baselines;
    num_coords = num_times * num_baselines;

    /* Initialise meta-data. */
    vis->freq_start_hz = 0.0;
    vis->freq_inc_hz = 0.0;
    vis->channel_bandwidth_hz = 0.0;
    vis->time_start_mjd_utc = 0.0;
    vis->time_inc_seconds = 0.0;
    vis->time_int_seconds = 0.0;
    vis->phase_centre_ra_deg = 0.0;
    vis->phase_centre_dec_deg = 0.0;

    /* Initialise memory. */
    oskar_mem_init(&vis->settings_path, OSKAR_CHAR, location, 0, 1, status);
    oskar_mem_init(&vis->telescope_path, OSKAR_CHAR, location, 0, 1, status);
    oskar_mem_init(&vis->x_metres, type, location, num_stations, 1, status);
    oskar_mem_init(&vis->y_metres, type, location, num_stations, 1, status);
    oskar_mem_init(&vis->z_metres, type, location, num_stations, 1, status);
    oskar_mem_init(&vis->receptor_angle, type, location, num_stations, 1, status);
    oskar_mem_init(&vis->uu_metres, type, location, num_coords, 1, status);
    oskar_mem_init(&vis->vv_metres, type, location, num_coords, 1, status);
    oskar_mem_init(&vis->ww_metres, type, location, num_coords, 1, status);
    oskar_mem_init(&vis->amplitude, amp_type, location, num_amps, 1, status);
}

#ifdef __cplusplus
}
#endif
