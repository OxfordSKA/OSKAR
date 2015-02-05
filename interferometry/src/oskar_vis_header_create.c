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

oskar_VisHeader* oskar_vis_header_create(int amp_type, int max_times_per_block,
        int num_times_total, int num_channels, int num_stations, int* status)
{
    oskar_VisHeader* hdr = 0;

    /* Check all inputs. */
    if (!status)
    {
        oskar_set_invalid_argument(status);
        return 0;
    }

    /* Check type. */
    if (!oskar_mem_type_is_complex(amp_type))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }

    /* Allocate the structure. */
    hdr = (oskar_VisHeader*) malloc(sizeof(oskar_VisHeader));
    hdr->amp_type = amp_type;

    /* Set dimensions. */
    hdr->max_times_per_block = max_times_per_block;
    hdr->num_times_total     = num_times_total;
    hdr->num_channels        = num_channels;
    hdr->num_stations        = num_stations;

    /* Initialise meta-data. */
    hdr->freq_start_hz = 0.0;
    hdr->freq_inc_hz = 0.0;
    hdr->channel_bandwidth_hz = 0.0;
    hdr->time_start_mjd_utc = 0.0;
    hdr->time_inc_sec = 0.0;
    hdr->time_average_sec = 0.0;
    hdr->phase_centre[0] = 0.0;
    hdr->phase_centre[1] = 0.0;
    hdr->telescope_centre[0] = 0.0; /* Longitude (deg). */
    hdr->telescope_centre[1] = 0.0; /* Latitude (deg). */
    hdr->telescope_centre[2] = 0.0; /* Altitude (m). */

    /* Initialise CPU memory. */
    hdr->telescope_path = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
    hdr->settings = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
    hdr->station_x_offset_ecef_metres = oskar_mem_create(OSKAR_DOUBLE,
            OSKAR_CPU, num_stations, status);
    hdr->station_y_offset_ecef_metres = oskar_mem_create(OSKAR_DOUBLE,
            OSKAR_CPU, num_stations, status);
    hdr->station_z_offset_ecef_metres = oskar_mem_create(OSKAR_DOUBLE,
            OSKAR_CPU, num_stations, status);

    /* Return handle to structure. */
    return hdr;
}

#ifdef __cplusplus
}
#endif
