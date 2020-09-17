/*
 * Copyright (c) 2017-2020, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include "imager/private_imager.h"
#include "imager/private_imager_read_coords.h"
#include "imager/oskar_imager.h"
#include "binary/oskar_binary.h"
#include "convert/oskar_convert_station_uvw_to_baseline_uvw.h"
#include "math/oskar_cmath.h"
#include "mem/oskar_binary_read_mem.h"
#include "ms/oskar_measurement_set.h"
#include "vis/oskar_vis_block.h"
#include "vis/oskar_vis_header.h"
#include "utility/oskar_timer.h"

#include <float.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

void oskar_imager_read_coords_ms(oskar_Imager* h, const char* filename,
        int i_file, int num_files, int* percent_done, int* percent_next,
        int* status)
{
#ifndef OSKAR_NO_MS
    oskar_MeasurementSet* ms;
    oskar_Mem *uvw, *u, *v, *w, *weight, *time_centroid;
    size_t start_row;
    double *uvw_, *u_, *v_, *w_;
    if (*status) return;

    /* Read the header. */
    oskar_log_message(h->log, 'M', 0, "Opening Measurement Set '%s'", filename);
    ms = oskar_ms_open(filename);
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

    /* Loop over visibility blocks. */
    for (start_row = 0; start_row < num_rows; start_row += num_baselines)
    {
        size_t allocated, required, block_size, i;
        if (*status) break;

        /* Read coordinates and weights from Measurement Set. */
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
                num_pols, u, v, w, 0, weight, time_centroid, status);
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


void oskar_imager_read_coords_vis(oskar_Imager* h, const char* filename,
        int i_file, int num_files, int* percent_done, int* percent_next,
        int* status)
{
    oskar_Binary* vis_file;
    oskar_VisHeader* hdr;
    oskar_Mem *u, *v, *w, *uu, *vv, *ww, *weight, *time_centroid;
    int i_block;
    double time_start_mjd, time_inc_sec;
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
    const int coord_prec = oskar_vis_header_coord_precision(hdr);
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
    u = oskar_mem_create(coord_prec, OSKAR_CPU, 0, status);
    v = oskar_mem_create(coord_prec, OSKAR_CPU, 0, status);
    w = oskar_mem_create(coord_prec, OSKAR_CPU, 0, status);
    uu = oskar_mem_create(coord_prec, OSKAR_CPU, 0, status);
    vv = oskar_mem_create(coord_prec, OSKAR_CPU, 0, status);
    ww = oskar_mem_create(coord_prec, OSKAR_CPU, 0, status);
    time_centroid = oskar_mem_create(OSKAR_DOUBLE,
            OSKAR_CPU, num_baselines * max_times_per_block, status);
    weight = oskar_mem_create(h->imager_prec, OSKAR_CPU, num_weights, status);
    oskar_mem_set_value_real(weight, 1.0, 0, num_weights, status);

    /* Loop over visibility blocks. */
    for (i_block = 0; i_block < num_blocks; ++i_block)
    {
        int c, t, dim_start_and_size[6], tag_error = 0;
        if (*status) break;

        /* Read block metadata. */
        oskar_timer_resume(h->tmr_read);
        oskar_binary_set_query_search_start(vis_file,
                i_block * tags_per_block, status);
        oskar_binary_read(vis_file, OSKAR_INT,
                OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_DIM_START_AND_SIZE, i_block,
                sizeof(dim_start_and_size), dim_start_and_size, status);
        const int start_time   = dim_start_and_size[0];
        const int start_chan   = dim_start_and_size[1];
        const int num_times    = dim_start_and_size[2];
        const int num_channels = dim_start_and_size[3];
        const size_t num_rows  = num_times * num_baselines;

        /* Fill in the time centroid values. */
        for (t = 0; t < num_times; ++t)
            oskar_mem_set_value_real(time_centroid,
                    time_start_mjd + (start_time + t + 0.5) * time_inc_sec,
                    t * num_baselines, num_baselines, status);

        /* Try to read station coordinates in the block. */
        oskar_binary_read_mem(vis_file, u, OSKAR_TAG_GROUP_VIS_BLOCK,
                OSKAR_VIS_BLOCK_TAG_STATION_U, i_block, &tag_error);
        if (!tag_error)
        {
            oskar_binary_read_mem(vis_file, v, OSKAR_TAG_GROUP_VIS_BLOCK,
                    OSKAR_VIS_BLOCK_TAG_STATION_V, i_block, status);
            oskar_binary_read_mem(vis_file, w, OSKAR_TAG_GROUP_VIS_BLOCK,
                    OSKAR_VIS_BLOCK_TAG_STATION_W, i_block, status);

            /* Convert from station to baseline coordinates. */
            for (t = 0; t < num_times; ++t)
                oskar_convert_station_uvw_to_baseline_uvw(num_stations,
                        num_stations * t, u, v, w,
                        num_baselines * t, uu, vv, ww, status);
        }
        else
        {
            /* Station coordinates not present,
             * so read the baseline coordinates directly. */
            oskar_binary_read_mem(vis_file, uu, OSKAR_TAG_GROUP_VIS_BLOCK,
                    OSKAR_VIS_BLOCK_TAG_BASELINE_UU, i_block, status);
            oskar_binary_read_mem(vis_file, vv, OSKAR_TAG_GROUP_VIS_BLOCK,
                    OSKAR_VIS_BLOCK_TAG_BASELINE_VV, i_block, status);
            oskar_binary_read_mem(vis_file, ww, OSKAR_TAG_GROUP_VIS_BLOCK,
                    OSKAR_VIS_BLOCK_TAG_BASELINE_WW, i_block, status);
        }

        /* Update the imager with the data. */
        oskar_timer_pause(h->tmr_read);
        for (c = 0; c < num_channels; ++c)
        {
            /* Update per channel. */
            const double freq_hz =
                    freq_start_hz + (start_chan + c) * freq_inc_hz;
            if (freq_hz >= h->freq_min_hz &&
                    (freq_hz <= h->freq_max_hz || h->freq_max_hz == 0.0))
            {
                oskar_imager_update(h, num_rows,
                        start_chan + c, start_chan + c, num_pols,
                        uu, vv, ww, 0, weight, time_centroid, status);
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
    oskar_mem_free(u, status);
    oskar_mem_free(v, status);
    oskar_mem_free(w, status);
    oskar_mem_free(uu, status);
    oskar_mem_free(vv, status);
    oskar_mem_free(ww, status);
    oskar_mem_free(weight, status);
    oskar_mem_free(time_centroid, status);
    oskar_vis_header_free(hdr, status);
    oskar_binary_free(vis_file);
}

#ifdef __cplusplus
}
#endif
