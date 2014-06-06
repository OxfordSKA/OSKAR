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

#include <private_vis.h>
#include <oskar_vis.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_Vis* oskar_vis_create_copy(const oskar_Vis* src, int location,
        int* status)
{
    oskar_Vis* dst = 0;

    /* Check all inputs. */
    if (!src || !status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Create the structure. */
    dst = oskar_vis_create(oskar_mem_type(oskar_vis_amplitude_const(src)),
            location, src->num_channels, src->num_times, src->num_stations,
            status);

    /* Copy the meta-data. */
    dst->num_channels  = src->num_channels;
    dst->num_times     = src->num_times;
    dst->num_stations  = src->num_stations;
    dst->num_baselines = src->num_baselines;
    dst->freq_start_hz = src->freq_start_hz;
    dst->freq_inc_hz = src->freq_inc_hz;
    dst->channel_bandwidth_hz = src->channel_bandwidth_hz;
    dst->time_start_mjd_utc = src->time_start_mjd_utc;
    dst->time_inc_sec = src->time_inc_sec;
    dst->time_average_sec = src->time_average_sec;
    dst->phase_centre_ra_deg = src->phase_centre_ra_deg;
    dst->phase_centre_dec_deg = src->phase_centre_dec_deg;
    dst->telescope_lon_deg = src->telescope_lon_deg;
    dst->telescope_lat_deg = src->telescope_lat_deg;
    dst->telescope_alt_metres = src->telescope_alt_metres;

    /* Copy the memory. */
    oskar_mem_copy(dst->settings_path, src->settings_path, status);
    oskar_mem_copy(dst->telescope_path, src->telescope_path, status);
    oskar_mem_copy(dst->settings, src->settings, status);
    oskar_mem_copy(dst->station_x_offset_ecef_metres,
            src->station_x_offset_ecef_metres, status);
    oskar_mem_copy(dst->station_y_offset_ecef_metres,
            src->station_y_offset_ecef_metres, status);
    oskar_mem_copy(dst->station_z_offset_ecef_metres,
            src->station_z_offset_ecef_metres, status);
    oskar_mem_copy(dst->station_x_enu_metres,
            src->station_x_enu_metres, status);
    oskar_mem_copy(dst->station_y_enu_metres,
            src->station_y_enu_metres, status);
    oskar_mem_copy(dst->station_z_enu_metres,
            src->station_z_enu_metres, status);
    oskar_mem_copy(dst->station_lon_deg, src->station_lon_deg, status);
    oskar_mem_copy(dst->station_lat_deg, src->station_lat_deg, status);
    oskar_mem_copy(dst->station_orientation_x_deg,
            src->station_orientation_x_deg, status);
    oskar_mem_copy(dst->station_orientation_y_deg,
            src->station_orientation_y_deg, status);
    oskar_mem_copy(dst->baseline_uu_metres, src->baseline_uu_metres, status);
    oskar_mem_copy(dst->baseline_vv_metres, src->baseline_vv_metres, status);
    oskar_mem_copy(dst->baseline_ww_metres, src->baseline_ww_metres, status);
    oskar_mem_copy(dst->amplitude, src->amplitude, status);

    /* Return a handle to the new structure. */
    return dst;
}

#ifdef __cplusplus
}
#endif
