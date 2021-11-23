/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "convert/oskar_convert_date_time_to_mjd.h"
#include "utility/oskar_dir.h"
#include "utility/oskar_get_error_string.h"
#include "vis/oskar_vis_block.h"
#include "vis/oskar_vis_header.h"

#include <cstdio>

TEST(write_ms, test_write)
{
    int status = 0;
    int num_antennas  = 5;
    int num_channels  = 3;
    int num_times     = 5;
    int num_baselines = num_antennas * (num_antennas - 1) / 2;
    int max_times_per_block = 2 * num_times;

    // Create a visibility structure and fill in some data.
    oskar_VisHeader* hdr = oskar_vis_header_create(OSKAR_DOUBLE_COMPLEX_MATRIX,
            OSKAR_DOUBLE, max_times_per_block, num_times, num_channels,
            num_channels, num_antennas, 1, 1, &status);
    oskar_VisBlock* blk = oskar_vis_block_create_from_header(OSKAR_CPU,
            hdr, &status);
    oskar_vis_block_set_num_times(blk, num_times, &status);
    double4c* v_ = oskar_mem_double4c(
            oskar_vis_block_cross_correlations(blk), &status);
    double *uu = 0, *vv = 0, *ww = 0, *x = 0, *y = 0, *z = 0;
    uu = oskar_mem_double(oskar_vis_block_baseline_uu_metres(blk), &status);
    vv = oskar_mem_double(oskar_vis_block_baseline_vv_metres(blk), &status);
    ww = oskar_mem_double(oskar_vis_block_baseline_ww_metres(blk), &status);
    x = oskar_mem_double(
            oskar_vis_header_station_offset_ecef_metres(hdr, 0), &status);
    y = oskar_mem_double(
            oskar_vis_header_station_offset_ecef_metres(hdr, 1), &status);
    z = oskar_mem_double(
            oskar_vis_header_station_offset_ecef_metres(hdr, 2), &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    for (int i = 0, t = 0; t < num_times; ++t)
    {
        for (int c = 0; c < num_channels; ++c)
        {
            for (int b = 0; b < num_baselines; ++b, ++i)
            {
                // XX
                v_[i].a.x = (double)c + 0.1;
                v_[i].a.y = 0.05;
                // XY
                v_[i].b.x = (double)t + 0.1;
                v_[i].b.y = 0.15;
                // YX
                v_[i].c.x = (double)b + 0.1;
                v_[i].c.y = 0.25;
                // YY
                v_[i].d.x = (double)i + 0.1;
                v_[i].d.y = 0.35;
            }
        }
    }
    for (int i = 0, t = 0; t < num_times; ++t)
    {
        for (int b = 0; b < num_baselines; ++b, ++i)
        {
            uu[i] = (double)t + 0.001;
            vv[i] = (double)b + 0.002;
            ww[i] = (double)i + 0.003;
        }
    }
    for (int i = 0; i < num_antennas; ++i)
    {
        x[i] = (double)i + 0.1;
        y[i] = (double)i + 0.2;
        z[i] = (double)i + 0.3;
    }
    oskar_vis_header_set_phase_centre(hdr, 0, 160.0, 89.0);
    oskar_vis_header_set_freq_start_hz(hdr, 222.22e6);
    oskar_vis_header_set_freq_inc_hz(hdr, 11.1e6);
    oskar_vis_header_set_time_start_mjd_utc(hdr,
            oskar_convert_date_time_to_mjd(2011, 11, 17, 0.0));
    oskar_vis_header_set_time_inc_sec(hdr, 1.0);

    const char filename[] = "temp_test_write_ms.ms";
    const char log_line[] = "Log line";
    oskar_MeasurementSet* ms = oskar_vis_header_write_ms(hdr, filename,
            OSKAR_FALSE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_vis_block_write_ms(blk, hdr, ms, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_ms_add_history(ms, "OSKAR_LOG", log_line, sizeof(log_line));
    oskar_vis_header_free(hdr, &status);
    oskar_vis_block_free(blk, &status);
    oskar_ms_close(ms);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_dir_remove(filename);
}


TEST(write_ms, test_write_partial)
{
    int status = 0;
    int num_antennas  = 5;
    int num_channels  = 8;
    int num_times     = 10;
    int num_baselines = num_antennas * (num_antennas - 1) / 2;
    int max_times_per_block = num_times / 2;
    int max_channels_per_block = 3;

    // Create a visibility structure and fill in some data.
    oskar_VisHeader* hdr = oskar_vis_header_create(
            OSKAR_DOUBLE_COMPLEX_MATRIX, OSKAR_DOUBLE,
            max_times_per_block, num_times,
            max_channels_per_block, num_channels, num_antennas, 1, 1, &status);
    oskar_VisBlock* blk = oskar_vis_block_create_from_header(OSKAR_CPU,
            hdr, &status);

    oskar_vis_block_set_start_channel_index(blk, 4);
    double4c* v_ = oskar_mem_double4c(
            oskar_vis_block_cross_correlations(blk), &status);
    double *uu = 0, *vv = 0, *ww = 0, *x = 0, *y = 0, *z = 0;
    uu = oskar_mem_double(oskar_vis_block_baseline_uu_metres(blk), &status);
    vv = oskar_mem_double(oskar_vis_block_baseline_vv_metres(blk), &status);
    ww = oskar_mem_double(oskar_vis_block_baseline_ww_metres(blk), &status);
    x = oskar_mem_double(
            oskar_vis_header_station_offset_ecef_metres(hdr, 0), &status);
    y = oskar_mem_double(
            oskar_vis_header_station_offset_ecef_metres(hdr, 1), &status);
    z = oskar_mem_double(
            oskar_vis_header_station_offset_ecef_metres(hdr, 2), &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    for (int i = 0, t = 0; t < max_times_per_block; ++t)
    {
        for (int c = 0; c < max_channels_per_block; ++c)
        {
            for (int b = 0; b < num_baselines; ++b, ++i)
            {
                // XX
                v_[i].a.x = (double)c + 0.1;
                v_[i].a.y = 0.05;
                // XY
                v_[i].b.x = (double)t + 0.1;
                v_[i].b.y = 0.15;
                // YX
                v_[i].c.x = (double)b + 0.1;
                v_[i].c.y = 0.25;
                // YY
                v_[i].d.x = (double)i + 0.1;
                v_[i].d.y = 0.35;
            }
        }
    }
    for (int i = 0, t = 0; t < max_times_per_block; ++t)
    {
        for (int b = 0; b < num_baselines; ++b, ++i)
        {
            uu[i] = (double)t + 0.001;
            vv[i] = (double)b + 0.002;
            ww[i] = (double)i + 0.003;
        }
    }
    for (int i = 0; i < num_antennas; ++i)
    {
        x[i] = (double)i + 0.1;
        y[i] = (double)i + 0.2;
        z[i] = (double)i + 0.3;
    }
    oskar_vis_header_set_phase_centre(hdr, 0, 160.0, 89.0);
    oskar_vis_header_set_freq_start_hz(hdr, 222.22e6);
    oskar_vis_header_set_freq_inc_hz(hdr, 11.1e6);
    oskar_vis_header_set_time_start_mjd_utc(hdr,
            oskar_convert_date_time_to_mjd(2011, 11, 17, 0.0));
    oskar_vis_header_set_time_inc_sec(hdr, 1.0);

    const char filename[] = "temp_test_write_ms_partial.ms";
    const char log_line[] = "Log line";
    oskar_MeasurementSet* ms = oskar_vis_header_write_ms(hdr, filename,
            OSKAR_FALSE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_vis_block_write_ms(blk, hdr, ms, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_ms_add_history(ms, "OSKAR_LOG", log_line, sizeof(log_line));

    // Clean up.
    oskar_vis_header_free(hdr, &status);
    oskar_vis_block_free(blk, &status);
    oskar_ms_close(ms);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_dir_remove(filename);
}

