/*
 * Copyright (c) 2011-2014, The University of Oxford
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

#include <private_vis.h>
#include <oskar_vis.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_vis_resize(oskar_Vis* vis, int num_channels,
        int num_times, int num_stations, int* status)
{
    int num_amps, num_coords, num_baselines;

    /* Check all inputs. */
    if (!vis || !status)
    {
        oskar_set_invalid_argument(status);
        return;
    }

    /* Check if safe to proceed. */
    if (*status) return;

    num_baselines = num_stations * (num_stations - 1) / 2;
    vis->num_stations  = num_stations;
    vis->num_channels  = num_channels;
    vis->num_times     = num_times;
    vis->num_baselines = num_baselines;
    num_amps   = num_channels * num_times * num_baselines;
    num_coords = num_times * num_baselines;

    oskar_mem_realloc(vis->station_x_offset_ecef_metres, num_stations, status);
    oskar_mem_realloc(vis->station_y_offset_ecef_metres, num_stations, status);
    oskar_mem_realloc(vis->station_z_offset_ecef_metres, num_stations, status);
    oskar_mem_realloc(vis->station_x_enu_metres, num_stations, status);
    oskar_mem_realloc(vis->station_y_enu_metres, num_stations, status);
    oskar_mem_realloc(vis->station_z_enu_metres, num_stations, status);
    oskar_mem_realloc(vis->station_lon_deg, num_stations, status);
    oskar_mem_realloc(vis->station_lat_deg, num_stations, status);
    oskar_mem_realloc(vis->station_orientation_x_deg, num_stations, status);
    oskar_mem_realloc(vis->station_orientation_y_deg, num_stations, status);
    oskar_mem_realloc(vis->baseline_uu_metres, num_coords, status);
    oskar_mem_realloc(vis->baseline_vv_metres, num_coords, status);
    oskar_mem_realloc(vis->baseline_ww_metres, num_coords, status);
    oskar_mem_realloc(vis->amplitude, num_amps, status);
}

#ifdef __cplusplus
}
#endif
