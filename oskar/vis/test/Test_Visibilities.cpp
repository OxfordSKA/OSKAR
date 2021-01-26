/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "vis/oskar_vis_header.h"
#include "vis/oskar_vis_block.h"
#include "utility/oskar_get_error_string.h"

#include <cstring>
#include <cstdio>
#include <cmath>

TEST(Visibilities, read_write)
{
    int status = 0;
    int num_channels           = 10;
    int num_times              = 77;
    int num_stations           = 20;
    int num_baselines          = num_stations * (num_stations - 1) / 2;
    int max_times_per_block    = 10;
    int max_channels_per_block = num_channels;
    double start_freq          = 200.0e6;
    double freq_inc            = 10.0e6;
    double time_start_mjd_utc  = 10.0;
    double time_inc_seconds    = 1.5;
    int precision              = OSKAR_DOUBLE;
    int amp_type               = precision | OSKAR_COMPLEX | OSKAR_MATRIX;
    const char* filename       = "temp_test_vis.dat";

    // Calculate number of visibility blocks required.
    int num_blocks = (num_times + max_times_per_block - 1) / max_times_per_block;

    // Write visibilities.
    {
        // Create the header.
        oskar_VisHeader* hdr = oskar_vis_header_create(amp_type, precision,
                max_times_per_block, num_times,
                max_channels_per_block, num_channels,
                num_stations, 0, 1, &status);
        oskar_vis_header_set_freq_start_hz(hdr, start_freq);
        oskar_vis_header_set_freq_inc_hz(hdr, freq_inc);
        oskar_vis_header_set_time_start_mjd_utc(hdr, time_start_mjd_utc);
        oskar_vis_header_set_time_inc_sec(hdr, time_inc_seconds);
        const char* name = "dummy";
        oskar_mem_append_raw(oskar_vis_header_telescope_path(hdr), name,
                OSKAR_CHAR, OSKAR_CPU, 1 + strlen(name), &status);
        ASSERT_STREQ(name, oskar_mem_char_const(
                oskar_vis_header_telescope_path_const(hdr)));
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(OSKAR_VIS_POL_TYPE_LINEAR_XX_XY_YX_YY,
                oskar_vis_header_pol_type(hdr));

        // Write the header.
        oskar_Binary* h = oskar_vis_header_write(hdr, filename, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Create a visibility block.
        oskar_VisBlock* blk = oskar_vis_block_create_from_header(
                OSKAR_CPU, hdr, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(num_baselines, oskar_vis_block_num_baselines(blk));

        // Get pointers to data in block.
        double4c* v_ = oskar_mem_double4c(
                oskar_vis_block_cross_correlations(blk), &status);
        double* uu = oskar_mem_double(
                oskar_vis_block_baseline_uu_metres(blk), &status);
        double* vv = oskar_mem_double(
                oskar_vis_block_baseline_vv_metres(blk), &status);
        double* ww = oskar_mem_double(
                oskar_vis_block_baseline_ww_metres(blk), &status);

        // Loop over blocks.
        for (int i_block = 0; i_block < num_blocks; ++i_block)
        {
            for (int i = 0, t = 0; t < max_times_per_block; ++t)
            {
                for (int c = 0; c < max_channels_per_block; ++c)
                {
                    for (int b = 0; b < num_baselines; ++b, ++i)
                    {
                        // XX
                        v_[i].a.x = 0.1 + (double)(t + i_block * max_times_per_block);
                        v_[i].a.y = 0.05;
                        // XY
                        v_[i].b.x = 0.2 + (double)c;
                        v_[i].b.y = 0.15;
                        // YX
                        v_[i].c.x = 0.3 + (double)b;
                        v_[i].c.y = 0.25;
                        // YY
                        v_[i].d.x = 0.4 + (double)i;
                        v_[i].d.y = 0.35;
                    }
                }
            }
            for (int i = 0, t = 0; t < max_times_per_block; ++t)
            {
                for (int b = 0; b < num_baselines; ++b, ++i)
                {
                    uu[i] = 0.1 + (double)(t + i_block * max_times_per_block);
                    vv[i] = 0.2 + (double)b;
                    ww[i] = 0.3 + (double)i;
                }
            }

            // Write the block.
            oskar_vis_block_write(blk, h, i_block, &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
        oskar_vis_header_free(hdr, &status);
        oskar_vis_block_free(blk, &status);
        oskar_binary_free(h);
    }

    // Read visibilities.
    {
        // Read the header.
        oskar_Binary* h = oskar_binary_create(filename, 'r', &status);
        oskar_VisHeader* hdr = oskar_vis_header_read(h, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(amp_type, oskar_vis_header_amp_type(hdr));
        ASSERT_EQ(precision, oskar_vis_header_coord_precision(hdr));
        ASSERT_EQ(num_channels, oskar_vis_header_num_channels_total(hdr));
        ASSERT_EQ(num_times, oskar_vis_header_num_times_total(hdr));
        ASSERT_EQ(start_freq, oskar_vis_header_freq_start_hz(hdr));
        ASSERT_EQ(freq_inc, oskar_vis_header_freq_inc_hz(hdr));
        ASSERT_EQ(time_start_mjd_utc, oskar_vis_header_time_start_mjd_utc(hdr));
        ASSERT_EQ(time_inc_seconds, oskar_vis_header_time_inc_sec(hdr));

        // Create a visibility block.
        oskar_VisBlock* blk = oskar_vis_block_create_from_header(
                OSKAR_CPU, hdr, &status);

        // Loop over blocks.
        for (int i_block = 0; i_block < num_blocks; ++i_block)
        {
            // Read the block.
            oskar_vis_block_read(blk, hdr, h, i_block, &status);
            ASSERT_EQ(num_baselines, oskar_vis_block_num_baselines(blk));

            // Check the data loaded correctly.
            double4c* v_ = oskar_mem_double4c(
                    oskar_vis_block_cross_correlations(blk), &status);
            double* uu = oskar_mem_double(
                    oskar_vis_block_baseline_uu_metres(blk), &status);
            double* vv = oskar_mem_double(
                    oskar_vis_block_baseline_vv_metres(blk), &status);
            double* ww = oskar_mem_double(
                    oskar_vis_block_baseline_ww_metres(blk), &status);
            for (int i = 0, t = 0; t < max_times_per_block; ++t)
            {
                for (int c = 0; c < max_channels_per_block; ++c)
                {
                    for (int b = 0; b < num_baselines; ++b, ++i)
                    {
                        // XX
                        ASSERT_DOUBLE_EQ(0.1 + (double)(t + i_block * max_times_per_block), v_[i].a.x);
                        ASSERT_DOUBLE_EQ(0.05, v_[i].a.y);
                        // XY
                        ASSERT_DOUBLE_EQ(0.2 + (double)c, v_[i].b.x);
                        ASSERT_DOUBLE_EQ(0.15, v_[i].b.y);
                        // YX
                        ASSERT_DOUBLE_EQ(0.3 + (double)b, v_[i].c.x);
                        ASSERT_DOUBLE_EQ(0.25, v_[i].c.y);
                        // YY
                        ASSERT_DOUBLE_EQ(0.4 + (double)i, v_[i].d.x);
                        ASSERT_DOUBLE_EQ(0.35, v_[i].d.y);
                    }
                }
            }
            for (int i = 0, t = 0; t < max_times_per_block; ++t)
            {
                for (int b = 0; b < num_baselines; ++b, ++i)
                {
                    ASSERT_DOUBLE_EQ(0.1 + (double)(t + i_block * max_times_per_block), uu[i]);
                    ASSERT_DOUBLE_EQ(0.2 + (double)b, vv[i]);
                    ASSERT_DOUBLE_EQ(0.3 + (double)i, ww[i]);
                }
            }
        }
        oskar_vis_header_free(hdr, &status);
        oskar_vis_block_free(blk, &status);
        oskar_binary_free(h);
    }

    // Delete temporary file.
    remove(filename);
}
