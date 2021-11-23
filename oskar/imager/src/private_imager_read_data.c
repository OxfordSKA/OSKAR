/*
 * Copyright (c) 2017-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"
#include "imager/private_imager_read_data.h"
#include "imager/oskar_imager.h"
#include "binary/oskar_binary.h"
#include "math/oskar_cmath.h"
#include "mem/oskar_binary_read_mem.h"
#include "ms/oskar_measurement_set.h"
#include "utility/oskar_timer.h"
#include "vis/oskar_vis_block.h"
#include "vis/oskar_vis_header.h"

#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_read_data_ms(oskar_Imager* h, const char* filename,
        int i_file, int num_files, int* percent_done, int* percent_next,
        int* status)
{
#ifndef OSKAR_NO_MS
    oskar_MeasurementSet* ms = 0;
    oskar_Mem *uvw = 0, *u = 0, *v = 0, *w = 0, *data = 0;
    oskar_Mem *weight = 0, *time_centroid = 0;
    int type = 0;
    size_t start_row = 0;
    double *uvw_ = 0, *u_ = 0, *v_ = 0, *w_ = 0;
    if (*status) return;

    /* Read the header. */
    oskar_log_message(h->log, 'M', 0, "Opening Measurement Set '%s'", filename);
    ms = oskar_ms_open_readonly(filename);
    if (!ms)
    {
        *status = OSKAR_ERR_FILE_IO;
        return;
    }
    const size_t num_rows = (size_t) oskar_ms_num_rows(ms);
    const size_t num_stations = (size_t) oskar_ms_num_stations(ms);
    const size_t num_baselines = num_stations * (num_stations - 1) / 2;
    const int num_pols = (int) oskar_ms_num_pols(ms);
    const int num_channels = (int) oskar_ms_num_channels(ms);

    /* Set visibility meta-data. */
    oskar_imager_set_vis_frequency(h,
            oskar_ms_freq_start_hz(ms),
            oskar_ms_freq_inc_hz(ms), num_channels);
    oskar_imager_set_vis_phase_centre(h,
            oskar_ms_phase_centre_ra_rad(ms) * 180/M_PI,
            oskar_ms_phase_centre_dec_rad(ms) * 180/M_PI);

    /* Create arrays. */
    uvw = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 3 * num_baselines, status);
    u = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, num_baselines, status);
    v = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, num_baselines, status);
    w = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, num_baselines, status);
    weight = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU,
            num_baselines * num_pols, status);
    time_centroid = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, num_baselines,
            status);
    uvw_ = oskar_mem_double(uvw, status);
    u_ = oskar_mem_double(u, status);
    v_ = oskar_mem_double(v, status);
    w_ = oskar_mem_double(w, status);
    type = OSKAR_SINGLE | OSKAR_COMPLEX;
    if (num_pols == 4) type |= OSKAR_MATRIX;
    data = oskar_mem_create(type, OSKAR_CPU,
            num_baselines * num_channels, status);

    /* Loop over visibility blocks. */
    for (start_row = 0; start_row < num_rows; start_row += num_baselines)
    {
        size_t allocated = 0, required = 0, block_size = 0, i = 0;
        if (*status) break;

        /* Read rows from Measurement Set. */
        oskar_timer_resume(h->tmr_read);
        block_size = num_rows - start_row;
        if (block_size > num_baselines) block_size = num_baselines;
        allocated = oskar_mem_length(uvw) *
                oskar_mem_element_size(oskar_mem_type(uvw));
        oskar_ms_read_column(ms, "UVW", start_row, block_size,
                allocated, oskar_mem_void(uvw), &required, status);
        allocated = oskar_mem_length(weight) *
                oskar_mem_element_size(oskar_mem_type(weight));
        oskar_ms_read_column(ms, "WEIGHT", start_row, block_size,
                allocated, oskar_mem_void(weight), &required, status);
        allocated = oskar_mem_length(time_centroid) *
                oskar_mem_element_size(oskar_mem_type(time_centroid));
        oskar_ms_read_column(ms, "TIME_CENTROID", start_row, block_size,
                allocated, oskar_mem_void(time_centroid), &required, status);
        allocated = oskar_mem_length(data) *
                oskar_mem_element_size(oskar_mem_type(data));
        oskar_ms_read_column(ms, h->ms_column, start_row, block_size,
                allocated, oskar_mem_void(data), &required, status);
        if (*status) break;

        /* Split up baseline coordinates. */
        for (i = 0; i < block_size; ++i)
        {
            u_[i] = uvw_[3*i + 0];
            v_[i] = uvw_[3*i + 1];
            w_[i] = uvw_[3*i + 2];
        }

        /* Update the imager with the data. */
        oskar_timer_pause(h->tmr_read);
        oskar_imager_update(h, block_size, 0, num_channels - 1,
                num_pols, u, v, w, data, weight, time_centroid, status);
        *percent_done = (int) round(100.0 * (
                (start_row + block_size) / (double)(num_rows * num_files) +
                i_file / (double)num_files));
        if (percent_next && *percent_done >= *percent_next)
        {
            oskar_log_message(h->log, 'S', -2, "%3d%% ...", *percent_done);
            *percent_next = 10 + 10 * (*percent_done / 10);
        }
    }
    oskar_mem_free(uvw, status);
    oskar_mem_free(u, status);
    oskar_mem_free(v, status);
    oskar_mem_free(w, status);
    oskar_mem_free(data, status);
    oskar_mem_free(weight, status);
    oskar_mem_free(time_centroid, status);
    oskar_ms_close(ms);
#else
    (void) filename;
    (void) i_file;
    (void) num_files;
    (void) percent_done;
    (void) percent_next;
    oskar_log_error(h->log,
            "OSKAR was compiled without Measurement Set support.");
    *status = OSKAR_ERR_FUNCTION_NOT_AVAILABLE;
#endif
}


void oskar_imager_read_data_vis(oskar_Imager* h, const char* filename,
        int i_file, int num_files, int* percent_done, int* percent_next,
        int* status)
{
    oskar_Binary* vis_file = 0;
    oskar_VisBlock* block = 0;
    oskar_VisHeader* hdr = 0;
    oskar_Mem *weight = 0, *time_centroid = 0, *scratch = 0;
    int i_block = 0;
    double time_start_mjd = 0.0, time_inc_sec = 0.0;
    if (*status) return;

    /* Read the header. */
    oskar_log_message(h->log, 'M', 0, "Opening '%s'", filename);
    vis_file = oskar_binary_create(filename, 'r', status);
    hdr = oskar_vis_header_read(vis_file, status);
    if (*status)
    {
        oskar_vis_header_free(hdr, status);
        oskar_binary_free(vis_file);
        return;
    }
    const int max_times_per_block = oskar_vis_header_max_times_per_block(hdr);
    const int tags_per_block = oskar_vis_header_num_tags_per_block(hdr);
    const int num_stations = oskar_vis_header_num_stations(hdr);
    const int num_baselines = num_stations * (num_stations - 1) / 2;
    const int num_pols =
            oskar_type_is_matrix(oskar_vis_header_amp_type(hdr)) ? 4 : 1;
    const int num_weights = num_baselines * num_pols * max_times_per_block;
    const int num_blocks = oskar_vis_header_num_blocks(hdr);
    const double freq_inc_hz = oskar_vis_header_freq_inc_hz(hdr);
    const double freq_start_hz = oskar_vis_header_freq_start_hz(hdr);
    time_start_mjd = oskar_vis_header_time_start_mjd_utc(hdr) * 86400.0;
    time_inc_sec = oskar_vis_header_time_inc_sec(hdr);

    /* Set visibility meta-data. */
    oskar_imager_set_vis_frequency(h, freq_start_hz, freq_inc_hz,
            oskar_vis_header_num_channels_total(hdr));
    oskar_imager_set_vis_phase_centre(h,
            oskar_vis_header_phase_centre_ra_deg(hdr),
            oskar_vis_header_phase_centre_dec_deg(hdr));

    /* Create scratch arrays. Weights are all 1. */
    time_centroid = oskar_mem_create(OSKAR_DOUBLE,
            OSKAR_CPU, num_baselines * max_times_per_block, status);
    weight = oskar_mem_create(h->imager_prec, OSKAR_CPU, num_weights, status);
    oskar_mem_set_value_real(weight, 1.0, 0, num_weights, status);
    scratch = oskar_mem_create(oskar_vis_header_amp_type(hdr), OSKAR_CPU,
            num_baselines * max_times_per_block, status);

    /* Loop over visibility blocks. */
    block = oskar_vis_block_create_from_header(OSKAR_CPU, hdr, status);
    for (i_block = 0; i_block < num_blocks; ++i_block)
    {
        int c = 0, t = 0;
        if (*status) break;

        /* Read the visibility data. */
        oskar_timer_resume(h->tmr_read);
        oskar_binary_set_query_search_start(vis_file,
                i_block * tags_per_block, status);
        oskar_vis_block_read(block, hdr, vis_file, i_block, status);
        const int start_time   = oskar_vis_block_start_time_index(block);
        const int start_chan   = oskar_vis_block_start_channel_index(block);
        const int num_times    = oskar_vis_block_num_times(block);
        const int num_channels = oskar_vis_block_num_channels(block);
        const size_t num_rows  = num_times * num_baselines;

        /* Fill in the time centroid values. */
        for (t = 0; t < num_times; ++t)
        {
            oskar_mem_set_value_real(time_centroid,
                    time_start_mjd + (start_time + t + 0.5) * time_inc_sec,
                    t * num_baselines, num_baselines, status);
        }
        oskar_timer_pause(h->tmr_read);

        /* Update the imager with the data. */
        for (c = 0; c < num_channels; ++c)
        {
            /* Update per channel. */
            const double freq_hz =
                    freq_start_hz + (start_chan + c) * freq_inc_hz;
            if (freq_hz >= h->freq_min_hz &&
                    (freq_hz <= h->freq_max_hz || h->freq_max_hz == 0.0))
            {
                oskar_timer_resume(h->tmr_copy_convert);
                for (t = 0; t < num_times; ++t)
                {
                    oskar_mem_copy_contents(scratch,
                            oskar_vis_block_cross_correlations(block),
                            num_baselines * t,
                            num_baselines * (num_channels * t + c),
                            num_baselines, status);
                }
                oskar_timer_pause(h->tmr_copy_convert);
                oskar_imager_update(h, num_rows,
                        start_chan + c, start_chan + c, num_pols,
                        oskar_vis_block_baseline_uu_metres_const(block),
                        oskar_vis_block_baseline_vv_metres_const(block),
                        oskar_vis_block_baseline_ww_metres_const(block),
                        scratch, weight, time_centroid, status);
            }
        }
        *percent_done = (int) round(100.0 * (
                (i_block + 1) / (double)(num_blocks * num_files) +
                i_file / (double)num_files));
        if (percent_next && *percent_done >= *percent_next)
        {
            oskar_log_message(h->log, 'S', -2, "%3d%% ...", *percent_done);
            *percent_next = 10 + 10 * (*percent_done / 10);
        }
    }
    oskar_mem_free(scratch, status);
    oskar_mem_free(weight, status);
    oskar_mem_free(time_centroid, status);
    oskar_vis_block_free(block, status);
    oskar_vis_header_free(hdr, status);
    oskar_binary_free(vis_file);
}

#ifdef __cplusplus
}
#endif
