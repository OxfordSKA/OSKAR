/*
 * Copyright (c) 2015, The University of Oxford
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

#include <private_vis_header.h>
#include <oskar_vis_header.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_VisHeader* oskar_vis_header_create_copy(const oskar_VisHeader* other,
        int* status)
{
    oskar_VisHeader* hdr = 0;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Create a new header. */
    hdr = oskar_vis_header_create(other->amp_type, other->coord_precision,
            other->max_times_per_block, other->num_times_total,
            other->max_channels_per_block, other->num_channels_total,
            other->num_stations, other->write_autocorr, other->write_crosscorr,
            status);

    /* Copy meta-data. */
    hdr->pol_type = other->pol_type;
    hdr->freq_start_hz = other->freq_start_hz;
    hdr->freq_inc_hz = other->freq_inc_hz;
    hdr->channel_bandwidth_hz = other->channel_bandwidth_hz;
    hdr->time_start_mjd_utc = other->time_start_mjd_utc;
    hdr->time_inc_sec = other->time_inc_sec;
    hdr->time_average_sec = other->time_average_sec;
    hdr->phase_centre_type = other->phase_centre_type;
    hdr->phase_centre_deg[0] = other->phase_centre_deg[0];
    hdr->phase_centre_deg[1] = other->phase_centre_deg[1];
    hdr->telescope_centre_lon_deg = other->telescope_centre_lon_deg;
    hdr->telescope_centre_lat_deg = other->telescope_centre_lat_deg;
    hdr->telescope_centre_alt_m = other->telescope_centre_alt_m;

    /* Copy memory. */
    oskar_mem_copy(hdr->telescope_path, other->telescope_path, status);
    oskar_mem_copy(hdr->settings, other->settings, status);
    oskar_mem_copy(hdr->station_x_offset_ecef_metres,
            other->station_x_offset_ecef_metres, status);
    oskar_mem_copy(hdr->station_y_offset_ecef_metres,
            other->station_y_offset_ecef_metres, status);
    oskar_mem_copy(hdr->station_z_offset_ecef_metres,
            other->station_z_offset_ecef_metres, status);

    /* Return handle to structure. */
    return hdr;
}

#ifdef __cplusplus
}
#endif
