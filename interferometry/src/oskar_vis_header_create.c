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
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

oskar_VisHeader* oskar_vis_header_create(int amp_type, int coord_precision,
        int max_times_per_block, int num_times_total,
        int max_channels_per_block, int num_channels_total, int num_stations,
        int write_autocorr, int write_crosscor, int* status)
{
    oskar_VisHeader* hdr = 0;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Check type. */
    if (!oskar_type_is_complex(amp_type))
    {
        *status = OSKAR_ERR_BAD_DATA_TYPE;
        return 0;
    }

    /* Allocate the structure. */
    hdr = (oskar_VisHeader*) malloc(sizeof(oskar_VisHeader));
    if (!hdr)
    {
        *status = OSKAR_ERR_MEMORY_ALLOC_FAILURE;
        return 0;
    }
    hdr->amp_type = amp_type;
    hdr->coord_precision = coord_precision;

    /* Set number of tags per block in the binary file. */
    /* This must be updated if the number of fields written to file from
     * the oskar_VisBlock structure is changed. */
    hdr->num_tags_per_block = 1;
    if (write_crosscor) hdr->num_tags_per_block += 4;
    if (write_autocorr) hdr->num_tags_per_block += 1;

    /* Set dimensions. */
    hdr->max_times_per_block    = max_times_per_block;
    hdr->num_times_total        = num_times_total;
    hdr->max_channels_per_block = max_channels_per_block;
    hdr->num_channels_total     = num_channels_total;
    hdr->num_stations           = num_stations;

    /* Set default polarisation type. */
    if (oskar_type_is_matrix(amp_type))
        hdr->pol_type = OSKAR_VIS_POL_TYPE_LINEAR_XX_XY_YX_YY;
    else
        hdr->pol_type = OSKAR_VIS_POL_TYPE_STOKES_I;

    /* Initialise meta-data. */
    hdr->write_autocorr = write_autocorr;
    hdr->write_crosscorr = write_crosscor;
    hdr->freq_start_hz = 0.0;
    hdr->freq_inc_hz = 0.0;
    hdr->channel_bandwidth_hz = 0.0;
    hdr->time_start_mjd_utc = 0.0;
    hdr->time_inc_sec = 0.0;
    hdr->time_average_sec = 0.0;
    hdr->phase_centre_type = 0;
    hdr->phase_centre_deg[0] = 0.0;
    hdr->phase_centre_deg[1] = 0.0;
    hdr->telescope_centre_lon_deg = 0.0;
    hdr->telescope_centre_lat_deg = 0.0;
    hdr->telescope_centre_alt_m = 0.0;

    /* Initialise CPU memory. */
    hdr->telescope_path = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
    hdr->settings = oskar_mem_create(OSKAR_CHAR, OSKAR_CPU, 0, status);
    hdr->station_x_offset_ecef_metres = oskar_mem_create(coord_precision,
            OSKAR_CPU, num_stations, status);
    hdr->station_y_offset_ecef_metres = oskar_mem_create(coord_precision,
            OSKAR_CPU, num_stations, status);
    hdr->station_z_offset_ecef_metres = oskar_mem_create(coord_precision,
            OSKAR_CPU, num_stations, status);

    /* Return handle to structure. */
    return hdr;
}

#ifdef __cplusplus
}
#endif
