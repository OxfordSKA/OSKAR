/*
 * Copyright (c) 2011-2016, The University of Oxford
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

#include "vis/private_vis.h"
#include "vis/oskar_vis.h"
#include "binary/oskar_binary.h"
#include "log/oskar_log.h"
#include "vis/oskar_vis_header.h"
#include "vis/oskar_vis_block.h"

#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_vis_write(const oskar_Vis* vis, oskar_Log* log,
        const char* filename, int* status)
{
    int amp_type, coord_precision, i, num_baselines, num_blocks, num_channels;
    int num_stations, num_times;
    double freq_ref_hz, freq_inc_hz, time_ref_mjd_utc, time_inc_sec;
    oskar_Binary* h = 0;
    char* log_data = 0;
    size_t log_size = 0;
    oskar_VisHeader* hdr = 0;
    oskar_VisBlock* blk = 0;
    oskar_Mem* xcorr = 0;
    const oskar_Mem* amp = 0;
    int max_times_per_block = 10;

    /* Check if safe to proceed. */
    if (*status) return;

    /* Create a header. */
    amp_type = oskar_mem_type(oskar_vis_amplitude_const(vis));
    coord_precision = oskar_mem_type(oskar_vis_baseline_uu_metres_const(vis));
    num_channels = oskar_vis_num_channels(vis);
    num_stations = oskar_vis_num_stations(vis);
    num_times = oskar_vis_num_times(vis);
    hdr = oskar_vis_header_create(amp_type, coord_precision,
            max_times_per_block, num_times, num_channels, num_channels,
            num_stations, 0, 1, status);

    /* Copy station coordinates and metadata. */
    freq_ref_hz = oskar_vis_freq_start_hz(vis);
    freq_inc_hz = oskar_vis_freq_inc_hz(vis);
    time_ref_mjd_utc = oskar_vis_time_start_mjd_utc(vis);
    time_inc_sec = oskar_vis_time_inc_sec(vis);
    oskar_mem_copy(oskar_vis_header_station_x_offset_ecef_metres(hdr),
            oskar_vis_station_x_offset_ecef_metres_const(vis), status);
    oskar_mem_copy(oskar_vis_header_station_y_offset_ecef_metres(hdr),
            oskar_vis_station_y_offset_ecef_metres_const(vis), status);
    oskar_mem_copy(oskar_vis_header_station_z_offset_ecef_metres(hdr),
            oskar_vis_station_z_offset_ecef_metres_const(vis), status);
    oskar_mem_copy(oskar_vis_header_settings(hdr),
            oskar_vis_settings_const(vis), status);
    oskar_mem_copy(oskar_vis_header_telescope_path(hdr),
            oskar_vis_telescope_path_const(vis), status);
    oskar_vis_header_set_channel_bandwidth_hz(hdr,
            oskar_vis_channel_bandwidth_hz(vis));
    oskar_vis_header_set_freq_inc_hz(hdr, freq_inc_hz);
    oskar_vis_header_set_freq_start_hz(hdr, freq_ref_hz);
    oskar_vis_header_set_phase_centre(hdr, 0,
            oskar_vis_phase_centre_ra_deg(vis),
            oskar_vis_phase_centre_dec_deg(vis));
    oskar_vis_header_set_telescope_centre(hdr,
            oskar_vis_telescope_lon_deg(vis),
            oskar_vis_telescope_lat_deg(vis),
            oskar_vis_telescope_alt_metres(vis));
    oskar_vis_header_set_time_average_sec(hdr,
            oskar_vis_time_average_sec(vis));
    oskar_vis_header_set_time_inc_sec(hdr, time_inc_sec);
    oskar_vis_header_set_time_start_mjd_utc(hdr, time_ref_mjd_utc);

    /* Write the header. */
    h = oskar_vis_header_write(hdr, filename, status);

    /* Create a visibility block to copy into. */
    blk = oskar_vis_block_create_from_header(OSKAR_CPU, hdr, status);
    num_baselines = oskar_vis_block_num_baselines(blk);
    amp = oskar_vis_amplitude_const(vis);
    xcorr = oskar_vis_block_cross_correlations(blk);

    /* Work out the number of blocks. */
    num_blocks = (num_times + max_times_per_block - 1) / max_times_per_block;
    for (i = 0; i < num_blocks; ++i)
    {
        int block_length, time_offset, total_baselines, t, c;
        int block_start_time_index, block_end_time_index;

        /* Set up the block. */
        block_start_time_index = i * max_times_per_block;
        block_end_time_index = block_start_time_index +
                max_times_per_block - 1;
        if (block_end_time_index >= num_times)
            block_end_time_index = num_times - 1;
        block_length = 1 + block_end_time_index - block_start_time_index;
        oskar_vis_block_set_num_times(blk, block_length, status);
        oskar_vis_block_set_start_time_index(blk, block_start_time_index);

        /* Copy the baseline coordinate data. */
        time_offset = i * max_times_per_block * num_baselines;
        total_baselines = num_baselines * block_length;
        oskar_mem_copy_contents(oskar_vis_block_baseline_uu_metres(blk),
                oskar_vis_baseline_uu_metres_const(vis),
                0, time_offset, total_baselines, status);
        oskar_mem_copy_contents(oskar_vis_block_baseline_vv_metres(blk),
                oskar_vis_baseline_vv_metres_const(vis),
                0, time_offset, total_baselines, status);
        oskar_mem_copy_contents(oskar_vis_block_baseline_ww_metres(blk),
                oskar_vis_baseline_ww_metres_const(vis),
                0, time_offset, total_baselines, status);

        /* Fill the array from the old dimension order. */
        for (t = 0; t < block_length; ++t)
        {
            for (c = 0; c < num_channels; ++c)
            {
                oskar_mem_copy_contents(xcorr, amp,
                        num_baselines * (t * num_channels + c), num_baselines *
                        (c * num_times + i * max_times_per_block + t),
                        num_baselines, status);
            }
        }

        /* Write the block. */
        oskar_vis_block_write(blk, h, i, status);
    }

    /* If log exists, then write it out. */
    log_data = oskar_log_file_data(log, &log_size);
    if (log_data)
    {
        oskar_binary_write(h, OSKAR_CHAR, OSKAR_TAG_GROUP_RUN,
                OSKAR_TAG_RUN_LOG, 0, log_size, log_data, status);
        free(log_data);
    }

    /* Release the handles. */
    oskar_vis_header_free(hdr, status);
    oskar_vis_block_free(blk, status);
    oskar_binary_free(h);
}

#ifdef __cplusplus
}
#endif
