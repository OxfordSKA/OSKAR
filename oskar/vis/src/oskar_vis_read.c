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
#include "vis/oskar_vis_header.h"
#include "vis/oskar_vis_block.h"
#include "binary/oskar_binary.h"
#include "mem/oskar_binary_read_mem.h"

#include <math.h>

#ifdef __cplusplus
extern "C" {
#endif

static oskar_Vis* oskar_vis_read_new(oskar_Binary* h, int* status);

oskar_Vis* oskar_vis_read(oskar_Binary* h, int* status)
{
    /* Visibility metadata. */
    int num_channels = 0, num_times = 0, num_stations = 0, tag_error = 0;
    int amp_type = 0;
    unsigned char grp = OSKAR_TAG_GROUP_VISIBILITY;
    oskar_Vis* vis = 0;

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Read visibility dimensions. */
    oskar_binary_read_int(h, grp, OSKAR_VIS_TAG_NUM_CHANNELS, 0,
            &num_channels, status);
    if (*status == OSKAR_ERR_BINARY_TAG_NOT_FOUND)
    {
        /* Try to read a new format visibility file. */
        *status = 0;
        return oskar_vis_read_new(h, status);
    }
    else if (*status) return 0;
    oskar_binary_read_int(h, grp, OSKAR_VIS_TAG_NUM_TIMES, 0,
            &num_times, status);
    oskar_binary_read_int(h, grp, OSKAR_VIS_TAG_NUM_STATIONS, 0,
            &num_stations, &tag_error);
    if (tag_error == OSKAR_ERR_BINARY_TAG_NOT_FOUND)
    {
        /* Check for number of baselines if number of stations not present. */
        int num_baselines = 0;
        oskar_binary_read_int(h, grp, OSKAR_VIS_TAG_NUM_BASELINES, 0,
                &num_baselines, status);

        /* Convert baselines to stations (care using floating point here). */
        num_stations = (int) floor(0.5 +
                (1.0 + sqrt(1.0 + 8.0 * num_baselines)) / 2.0);
    }
    else if (tag_error) *status = tag_error;
    oskar_binary_read_int(h, grp, OSKAR_VIS_TAG_AMP_TYPE, 0, &amp_type, status);

    /* Check if safe to proceed. */
    if (*status) return 0;

    /* Create the visibility structure. */
    vis = oskar_vis_create(amp_type, OSKAR_CPU, num_channels, num_times,
            num_stations, status);

    /* Optionally read the settings path (ignore the error code). */
    tag_error = 0;
    oskar_binary_read_mem(h, vis->settings_path,
            OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS_PATH, 0, &tag_error);

    /* Optionally read the settings data (ignore the error code). */
    tag_error = 0;
    oskar_binary_read_mem(h, vis->settings,
            OSKAR_TAG_GROUP_SETTINGS, OSKAR_TAG_SETTINGS, 0, &tag_error);

    /* Optionally read the telescope model path (ignore the error code). */
    tag_error = 0;
    oskar_binary_read_mem(h, vis->telescope_path,
            grp, OSKAR_VIS_TAG_TELESCOPE_PATH, 0, &tag_error);

    /* Read visibility metadata. */
    oskar_binary_read_double(h, grp, OSKAR_VIS_TAG_FREQ_START_HZ, 0,
            &vis->freq_start_hz, status);
    oskar_binary_read_double(h, grp, OSKAR_VIS_TAG_FREQ_INC_HZ, 0,
            &vis->freq_inc_hz, status);
    oskar_binary_read_double(h, grp, OSKAR_VIS_TAG_TIME_START_MJD_UTC, 0,
            &vis->time_start_mjd_utc, status);
    oskar_binary_read_double(h, grp, OSKAR_VIS_TAG_TIME_INC_SEC, 0,
            &vis->time_inc_sec, status);
    oskar_binary_read_double(h, grp, OSKAR_VIS_TAG_PHASE_CENTRE_RA, 0,
            &vis->phase_centre_ra_deg, status);
    oskar_binary_read_double(h, grp, OSKAR_VIS_TAG_PHASE_CENTRE_DEC, 0,
            &vis->phase_centre_dec_deg, status);

    /* Read the baseline coordinate arrays. */
    oskar_binary_read_mem(h, vis->baseline_uu_metres, grp,
            OSKAR_VIS_TAG_BASELINE_UU, 0, status);
    oskar_binary_read_mem(h, vis->baseline_vv_metres, grp,
            OSKAR_VIS_TAG_BASELINE_VV, 0, status);
    oskar_binary_read_mem(h, vis->baseline_ww_metres, grp,
            OSKAR_VIS_TAG_BASELINE_WW, 0, status);

    /* Read the visibility data. */
    oskar_binary_read_mem(h, vis->amplitude, grp,
            OSKAR_VIS_TAG_AMPLITUDE, 0, status);

    /* Optionally read station coordinates (ignore the error code). */
    tag_error = 0;
    oskar_binary_read_mem(h, vis->station_x_offset_ecef_metres,
            grp, OSKAR_VIS_TAG_STATION_X_OFFSET_ECEF, 0, &tag_error);
    oskar_binary_read_mem(h, vis->station_y_offset_ecef_metres,
            grp, OSKAR_VIS_TAG_STATION_Y_OFFSET_ECEF, 0, &tag_error);
    oskar_binary_read_mem(h, vis->station_z_offset_ecef_metres,
            grp, OSKAR_VIS_TAG_STATION_Z_OFFSET_ECEF, 0, &tag_error);

    /* Optionally read telescope lon., lat., alt. (ignore the error code) */
    tag_error = 0;
    oskar_binary_read_double(h, grp, OSKAR_VIS_TAG_TELESCOPE_LON, 0,
            &vis->telescope_lon_deg, &tag_error);
    oskar_binary_read_double(h, grp, OSKAR_VIS_TAG_TELESCOPE_LAT, 0,
            &vis->telescope_lat_deg, &tag_error);
    oskar_binary_read_double(h, grp, OSKAR_VIS_TAG_TELESCOPE_ALT, 0,
            &vis->telescope_alt_metres, &tag_error);

    /* Optionally read the channel bandwidth value. */
    tag_error = 0;
    oskar_binary_read_double(h, grp, OSKAR_VIS_TAG_CHANNEL_BANDWIDTH_HZ, 0,
            &vis->channel_bandwidth_hz, &tag_error);

    /* Optionally read the time integration value. */
    tag_error = 0;
    oskar_binary_read_double(h, grp, OSKAR_VIS_TAG_TIME_AVERAGE_SEC, 0,
            &vis->time_average_sec, &tag_error);

    /* Return a handle to the new structure. */
    return vis;
}

/*
 * Translation layer from new file format to old structure.
 * This will be deleted when the old oskar_Vis structure is fully retired.
 */
oskar_Vis* oskar_vis_read_new(oskar_Binary* h, int* status)
{
    oskar_VisHeader* hdr = 0;
    oskar_VisBlock* blk = 0;
    oskar_Vis* vis = 0;
    oskar_Mem* amp = 0;
    const oskar_Mem* xcorr = 0;
    int amp_type, max_times_per_block, num_channels, num_stations, num_times;
    int i, num_blocks;
    double freq_ref_hz, freq_inc_hz, time_ref_mjd_utc, time_inc_sec;

    /* Try to read the new header. */
    hdr = oskar_vis_header_read(h, status);
    if (*status)
    {
        oskar_vis_header_free(hdr, status);
        return 0;
    }

    /* Create the old vis structure. */
    amp_type = oskar_vis_header_amp_type(hdr);
    max_times_per_block = oskar_vis_header_max_times_per_block(hdr);
    num_channels = oskar_vis_header_num_channels_total(hdr);
    num_stations = oskar_vis_header_num_stations(hdr);
    num_times = oskar_vis_header_num_times_total(hdr);
    vis = oskar_vis_create(amp_type, OSKAR_CPU, num_channels, num_times,
            num_stations, status);
    if (*status)
    {
        oskar_vis_header_free(hdr, status);
        oskar_vis_free(vis, status);
        return 0;
    }

    /* Copy station coordinates and metadata. */
    freq_ref_hz = oskar_vis_header_freq_start_hz(hdr);
    freq_inc_hz = oskar_vis_header_freq_inc_hz(hdr);
    time_ref_mjd_utc = oskar_vis_header_time_start_mjd_utc(hdr);
    time_inc_sec = oskar_vis_header_time_inc_sec(hdr);
    oskar_mem_copy(oskar_vis_station_x_offset_ecef_metres(vis),
            oskar_vis_header_station_x_offset_ecef_metres_const(hdr), status);
    oskar_mem_copy(oskar_vis_station_y_offset_ecef_metres(vis),
            oskar_vis_header_station_y_offset_ecef_metres_const(hdr), status);
    oskar_mem_copy(oskar_vis_station_z_offset_ecef_metres(vis),
            oskar_vis_header_station_z_offset_ecef_metres_const(hdr), status);
    oskar_mem_copy(oskar_vis_settings(vis),
            oskar_vis_header_settings_const(hdr), status);
    oskar_mem_copy(oskar_vis_telescope_path(vis),
            oskar_vis_header_telescope_path_const(hdr), status);
    oskar_vis_set_channel_bandwidth_hz(vis,
            oskar_vis_header_channel_bandwidth_hz(hdr));
    oskar_vis_set_freq_inc_hz(vis, freq_inc_hz);
    oskar_vis_set_freq_start_hz(vis, freq_ref_hz);
    oskar_vis_set_phase_centre(vis,
            oskar_vis_header_phase_centre_ra_deg(hdr),
            oskar_vis_header_phase_centre_dec_deg(hdr));
    oskar_vis_set_telescope_position(vis,
            oskar_vis_header_telescope_lon_deg(hdr),
            oskar_vis_header_telescope_lat_deg(hdr),
            oskar_vis_header_telescope_alt_metres(hdr));
    oskar_vis_set_time_average_sec(vis,
            oskar_vis_header_time_average_sec(hdr));
    oskar_vis_set_time_inc_sec(vis, time_inc_sec);
    oskar_vis_set_time_start_mjd_utc(vis, time_ref_mjd_utc);

    /* Create a visibility block to read into. */
    blk = oskar_vis_block_create_from_header(OSKAR_CPU, hdr, status);
    amp = oskar_vis_amplitude(vis);
    xcorr = oskar_vis_block_cross_correlations_const(blk);

    /* Work out the number of blocks. */
    num_blocks = (num_times + max_times_per_block - 1) / max_times_per_block;
    for (i = 0; i < num_blocks; ++i)
    {
        int block_length, num_baselines, time_offset, total_baselines, t, c;

        /* Read the block. */
        oskar_vis_block_read(blk, hdr, h, i, status);
        num_baselines = oskar_vis_block_num_baselines(blk);
        block_length = oskar_vis_block_num_times(blk);

        /* Copy the baseline coordinate data. */
        time_offset = i * max_times_per_block * num_baselines;
        total_baselines = num_baselines * block_length;
        oskar_mem_copy_contents(oskar_vis_baseline_uu_metres(vis),
                oskar_vis_block_baseline_uu_metres_const(blk),
                time_offset, 0, total_baselines, status);
        oskar_mem_copy_contents(oskar_vis_baseline_vv_metres(vis),
                oskar_vis_block_baseline_vv_metres_const(blk),
                time_offset, 0, total_baselines, status);
        oskar_mem_copy_contents(oskar_vis_baseline_ww_metres(vis),
                oskar_vis_block_baseline_ww_metres_const(blk),
                time_offset, 0, total_baselines, status);

        /* Fill the array in the old dimension order. */
        for (t = 0; t < block_length; ++t)
        {
            for (c = 0; c < num_channels; ++c)
            {
                oskar_mem_copy_contents(amp, xcorr, num_baselines *
                        (c * num_times + i * max_times_per_block + t),
                        num_baselines * (t * num_channels + c),
                        num_baselines, status);
            }
        }
    }

    /* Clean up and return. */
    oskar_vis_block_free(blk, status);
    oskar_vis_header_free(hdr, status);
    return vis;
}


#ifdef __cplusplus
}
#endif
