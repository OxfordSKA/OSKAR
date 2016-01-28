/*
 * Copyright (c) 2016, The University of Oxford
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

#include <private_imager.h>

#include <oskar_binary.h>
#include <oskar_imager_set_options.h>
#include <oskar_imager_run.h>
#include <oskar_imager_finalise.h>
#include <oskar_imager_update.h>
#include <oskar_vis_block.h>
#include <oskar_vis_header.h>

#include <ms/oskar_measurement_set.h>

#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

static void oskar_imager_run_ms(oskar_Imager* h, const char* filename,
        int* status);
static void oskar_imager_run_vis(oskar_Imager* h, const char* filename,
        int* status);
static void oskar_imager_data_range(const int settings_range[2],
        int num_data_values, int range[2], int* status);

void oskar_imager_run(oskar_Imager* h, const char* filename, int* status)
{
    int len, use_ms;
    if (*status || !filename) return;

    /* Check filename for Measurement Set. */
    len = strlen(filename);
    if (len == 0) { *status = OSKAR_ERR_FILE_IO; return; }
    use_ms = (len >= 3) && (
            !strcmp(&filename[len-3], ".MS") ||
            !strcmp(&filename[len-3], ".ms") ) ? 1 : 0;
    if (use_ms)
        oskar_imager_run_ms(h, filename, status);
    else
        oskar_imager_run_vis(h, filename, status);

    /* Finalise the image plane(s) and write them out. */
    oskar_imager_finalise(h, status);
}

void oskar_imager_run_vis(oskar_Imager* h, const char* filename, int* status)
{
    oskar_Binary* vis_file;
    oskar_VisBlock* blk;
    oskar_VisHeader* hdr;
    int max_times_per_block, tags_per_block, i_block, num_blocks;
    int start_time, end_time, start_chan, end_chan;
    int num_times, num_channels, percent_done = 0, percent_next = 10;
    int dim_start_and_size[6];
    vis_file = oskar_binary_create(filename, 'r', status);
    hdr = oskar_vis_header_read(vis_file, status);
    if (*status)
    {
        oskar_vis_header_free(hdr, status);
        oskar_binary_free(vis_file);
        return;
    }
    max_times_per_block = oskar_vis_header_max_times_per_block(hdr);
    tags_per_block = oskar_vis_header_num_tags_per_block(hdr);
    oskar_imager_set_station_coords(h, oskar_vis_header_num_stations(hdr),
            oskar_vis_header_station_x_offset_ecef_metres_const(hdr),
            oskar_vis_header_station_y_offset_ecef_metres_const(hdr),
            oskar_vis_header_station_z_offset_ecef_metres_const(hdr),
            status);
    num_times = oskar_vis_header_num_times_total(hdr);
    num_channels = oskar_vis_header_num_channels_total(hdr);
    oskar_imager_data_range(h->chan_range, num_channels,
            h->vis_chan_range, status);
    oskar_imager_data_range(h->time_range, num_times,
            h->vis_time_range, status);
    if (*status)
    {
        oskar_vis_header_free(hdr, status);
        oskar_binary_free(vis_file);
        return;
    }

    /* Set visibility meta-data. */
    h->vis_freq_start_hz = oskar_vis_header_freq_start_hz(hdr);
    h->freq_inc_hz = oskar_vis_header_freq_inc_hz(hdr);
    h->vis_time_start_mjd_utc = oskar_vis_header_time_start_mjd_utc(hdr);
    h->time_inc_sec = oskar_vis_header_time_inc_sec(hdr);
    h->vis_centre_deg[0] = oskar_vis_header_phase_centre_ra_deg(hdr);
    h->vis_centre_deg[1] = oskar_vis_header_phase_centre_dec_deg(hdr);

    /* Loop over visibility blocks. */
    blk = oskar_vis_block_create(OSKAR_CPU, hdr, status);
    num_blocks = (num_times + max_times_per_block - 1) /
            max_times_per_block;
    printf(" |   0%%..."); fflush(stdout);
    for (i_block = 0; i_block < num_blocks; ++i_block)
    {
        if (*status) break;

        /* Read block metadata. */
        oskar_binary_set_query_search_start(vis_file,
                i_block * tags_per_block, status);
        oskar_binary_read(vis_file, OSKAR_INT,
                OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_DIM_START_AND_SIZE, i_block,
                sizeof(dim_start_and_size), dim_start_and_size, status);
        start_time = dim_start_and_size[0];
        start_chan = dim_start_and_size[1];
        end_time   = start_time + dim_start_and_size[2] - 1;
        end_chan   = start_chan + dim_start_and_size[3] - 1;

        /* Check that at least part of the block is in range. */
        if (end_time >= h->time_range[0] &&
                (start_time <= h->time_range[1] || h->time_range[1] < 0))
        {
            oskar_vis_block_read(blk, hdr, vis_file, i_block, status);
            oskar_imager_update(h, start_time, end_time, start_chan, end_chan,
                    oskar_vis_block_num_pols(blk),
                    oskar_vis_block_num_baselines(blk),
                    oskar_vis_block_baseline_uu_metres(blk),
                    oskar_vis_block_baseline_vv_metres(blk),
                    oskar_vis_block_baseline_ww_metres(blk),
                    oskar_vis_block_cross_correlations(blk), status);
        }

        /* Update progress. */
        percent_done = 100 * (i_block + 1) / (double)num_blocks;
        if (percent_done >= percent_next)
        {
            printf("%d%%", percent_next);
            if (percent_next < 100) printf("..."); else printf("\n |\n");
            fflush(stdout);
            percent_next += 10;
        }
    }
    oskar_vis_block_free(blk, status);
    oskar_vis_header_free(hdr, status);
    oskar_binary_free(vis_file);
}


void oskar_imager_run_ms(oskar_Imager* h, const char* filename, int* status)
{
#ifndef OSKAR_NO_MS
    oskar_MeasurementSet* ms;
    oskar_Mem *uvw, *u, *v, *w, *data;
    int num_rows, num_stations, num_baselines, num_pols;
    int start_time, end_time, start_chan, end_chan;
    int num_times, num_channels, percent_done = 0, percent_next = 10;
    int i, block_size, start_row, type;
    double *uvw_, *u_, *v_, *w_;
    ms = oskar_ms_open(filename);
    if (!ms)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    num_rows = (int) oskar_ms_num_rows(ms);
    num_stations = (int) oskar_ms_num_stations(ms);
    num_baselines = num_stations * (num_stations - 1) / 2;
    num_pols = (int) oskar_ms_num_pols(ms);
    num_channels = (int) oskar_ms_num_channels(ms);
    num_times = num_rows / num_baselines;
    start_time = end_time = 0;
    start_chan = 0; end_chan = num_channels - 1;

    /* Check for irregular data and override synthesis mode if required. */
    if (num_rows % num_baselines != 0)
    {
        printf(" | WARNING: Irregular data detected. "
                "Using full time synthesis.\n");
        oskar_imager_set_time_range(h, 0, -1, 0);
    }
    oskar_imager_data_range(h->chan_range, num_channels,
            h->vis_chan_range, status);
    oskar_imager_data_range(h->time_range, num_times,
            h->vis_time_range, status);
    if (*status)
    {
        oskar_ms_close(ms);
        return;
    }

    /* Set visibility meta-data. */
    h->vis_freq_start_hz = oskar_ms_ref_freq_hz(ms);
    h->freq_inc_hz = oskar_ms_channel_width_hz(ms);
    h->vis_time_start_mjd_utc = oskar_ms_start_time_mjd(ms);
    h->time_inc_sec = oskar_ms_time_inc_sec(ms);
    h->vis_centre_deg[0] = oskar_ms_phase_centre_ra_rad(ms) * 180/M_PI;
    h->vis_centre_deg[1] = oskar_ms_phase_centre_dec_rad(ms) * 180/M_PI;

    /* Create arrays. */
    uvw = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU,
            3 * num_baselines, status);
    u = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, num_baselines, status);
    v = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, num_baselines, status);
    w = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, num_baselines, status);
    uvw_ = oskar_mem_double(uvw, status);
    u_ = oskar_mem_double(u, status);
    v_ = oskar_mem_double(v, status);
    w_ = oskar_mem_double(w, status);
    type = OSKAR_SINGLE | OSKAR_COMPLEX;
    if (num_pols == 4) type |= OSKAR_MATRIX;
    data = oskar_mem_create(type, OSKAR_CPU,
            num_baselines * num_channels, status);

    /* Loop over visibility blocks. */
    printf(" |   0%%..."); fflush(stdout);
    for (start_row = 0; start_row < num_rows; start_row += num_baselines)
    {
        size_t allocated, required;
        if (*status) break;

        /* Read rows from Measurement Set. */
        block_size = num_rows - start_row;
        if (block_size > num_baselines) block_size = num_baselines;
        allocated = oskar_mem_length(uvw) *
                oskar_mem_element_size(oskar_mem_type(uvw));
        oskar_ms_get_column(ms, "UVW", start_row, block_size,
                allocated, oskar_mem_void(uvw), &required, status);
        allocated = oskar_mem_length(data) *
                oskar_mem_element_size(oskar_mem_type(data));
        oskar_ms_get_column(ms, h->ms_column, start_row, block_size,
                allocated, oskar_mem_void(data), &required, status);
        if (*status) break;

        /* TODO(FD) Swap baseline and channel dimensions. */
        if (num_channels != 1)
        {
            *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
            break;
        }

        /* Split up baseline coordinates. */
        for (i = 0; i < block_size; ++i)
        {
            u_[i] = uvw_[3*i + 0];
            v_[i] = uvw_[3*i + 1];
            w_[i] = uvw_[3*i + 2];
        }

        /* Add the baseline data. */
        oskar_imager_update(h, start_time, end_time, start_chan, end_chan,
                num_pols, num_baselines, u, v, w, data, status);
        start_time += 1;
        end_time += 1;

        /* Update progress. */
        percent_done = 100 * (start_row + num_baselines) / (double)num_rows;
        if (percent_done >= percent_next)
        {
            printf("%d%%", percent_next);
            if (percent_next < 100) printf("..."); else printf("\n |\n");
            fflush(stdout);
            percent_next += 10;
        }
    }
    oskar_mem_free(uvw, status);
    oskar_mem_free(u, status);
    oskar_mem_free(v, status);
    oskar_mem_free(w, status);
    oskar_mem_free(data, status);
    oskar_ms_close(ms);
#else
    fprintf(stderr, "ERROR: OSKAR was compiled without "
            "Measurement Set support.\n");
#endif
}


void oskar_imager_data_range(const int settings_range[2],
        int num_data_values, int range[2], int* status)
{
    if (*status) return;
    if (settings_range[0] >= num_data_values ||
            settings_range[1] >= num_data_values)
    {
        *status = OSKAR_ERR_INVALID_RANGE;
        return;
    }
    range[0] = settings_range[0] < 0 ? 0 : settings_range[0];
    range[1] = settings_range[1] < 0 ? num_data_values - 1 : settings_range[1];
    if (range[0] > range[1])
    {
        *status = OSKAR_ERR_INVALID_RANGE;
        return;
    }
}

#ifdef __cplusplus
}
#endif
