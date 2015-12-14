/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include <gtest/gtest.h>

#include <oskar_vis.h>
#include <oskar_get_error_string.h>

#include <cstring>
#include <iostream>
#include <cstdio>
#include <cmath>

TEST(Visibilities, create)
{
    int num_channels  = 4;
    int num_times     = 2;
    int num_stations  = 27;
    int num_baselines = num_stations * (num_stations - 1) / 2;
    int num_coords    = num_times * num_baselines;
    int num_amps      = num_channels * num_times * num_baselines;

    // Expect an error for non complex visibility data types.
    {
        int status = 0;
        oskar_Vis* vis;
        vis = oskar_vis_create(OSKAR_SINGLE, OSKAR_CPU,
                num_channels, num_times, num_stations, &status);
        EXPECT_EQ(OSKAR_ERR_BAD_DATA_TYPE, status);
        status = 0;
        oskar_vis_free(vis, &status);
        vis = oskar_vis_create(OSKAR_DOUBLE, OSKAR_CPU,
                num_channels, num_times, num_stations, &status);
        EXPECT_EQ(OSKAR_ERR_BAD_DATA_TYPE, status);
        status = 0;
        oskar_vis_free(vis, &status);
    }

    // Don't expect an error for complex types.
    {
        int location, status = 0;
        oskar_Vis* vis;
        location = OSKAR_CPU;
        vis = oskar_vis_create(OSKAR_DOUBLE_COMPLEX, location,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_vis_free(vis, &status);
        vis = oskar_vis_create(OSKAR_SINGLE_COMPLEX, location,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_vis_free(vis, &status);
        vis = oskar_vis_create(OSKAR_DOUBLE_COMPLEX_MATRIX, location,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_vis_free(vis, &status);
        vis = oskar_vis_create(OSKAR_SINGLE_COMPLEX_MATRIX, location,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_vis_free(vis, &status);
        location = OSKAR_GPU;
        vis = oskar_vis_create(OSKAR_DOUBLE_COMPLEX, location,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_vis_free(vis, &status);
        vis = oskar_vis_create(OSKAR_SINGLE_COMPLEX, location,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_vis_free(vis, &status);
        vis = oskar_vis_create(OSKAR_DOUBLE_COMPLEX_MATRIX, location,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_vis_free(vis, &status);
        vis = oskar_vis_create(OSKAR_SINGLE_COMPLEX_MATRIX, location,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_vis_free(vis, &status);
    }
    {
        int status = 0;

        // Construct visibility data on the CPU and check dimensions.
        oskar_Vis* vis;
        vis = oskar_vis_create(OSKAR_SINGLE_COMPLEX_MATRIX,
                OSKAR_CPU, num_channels, num_times, num_stations,
                &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ((int)OSKAR_CPU, oskar_vis_location(vis));

        ASSERT_EQ((int)OSKAR_SINGLE_COMPLEX_MATRIX,
                oskar_mem_type(oskar_vis_amplitude(vis)));
        ASSERT_EQ((int)OSKAR_SINGLE,
                oskar_mem_type(oskar_vis_baseline_uu_metres(vis)));
        ASSERT_EQ((int)OSKAR_SINGLE,
                oskar_mem_type(oskar_vis_baseline_vv_metres(vis)));
        ASSERT_EQ((int)OSKAR_SINGLE,
                oskar_mem_type(oskar_vis_baseline_ww_metres(vis)));
        ASSERT_EQ(num_channels, oskar_vis_num_channels(vis));
        ASSERT_EQ(num_times, oskar_vis_num_times(vis));
        ASSERT_EQ(num_baselines, oskar_vis_num_baselines(vis));
        ASSERT_EQ(num_stations, oskar_vis_num_stations(vis));
        ASSERT_EQ(num_amps, oskar_vis_num_channels(vis) *
                oskar_vis_num_times(vis) * oskar_vis_num_baselines(vis));
        ASSERT_EQ(num_amps, (int)oskar_mem_length(oskar_vis_amplitude(vis)));
        ASSERT_EQ(num_coords,
                (int)oskar_mem_length(oskar_vis_baseline_uu_metres(vis)));
        ASSERT_EQ(num_coords,
                (int)oskar_mem_length(oskar_vis_baseline_vv_metres(vis)));
        ASSERT_EQ(num_coords,
                (int)oskar_mem_length(oskar_vis_baseline_ww_metres(vis)));
        oskar_vis_free(vis, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        vis = oskar_vis_create(OSKAR_DOUBLE_COMPLEX_MATRIX,
                OSKAR_CPU, 0, 0, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ((int)OSKAR_CPU, oskar_vis_location(vis));
        ASSERT_EQ((int)OSKAR_DOUBLE_COMPLEX_MATRIX,
                oskar_mem_type(oskar_vis_amplitude(vis)));
        ASSERT_EQ((int)OSKAR_DOUBLE,
                oskar_mem_type(oskar_vis_baseline_uu_metres(vis)));
        ASSERT_EQ((int)OSKAR_DOUBLE,
                oskar_mem_type(oskar_vis_baseline_vv_metres(vis)));
        ASSERT_EQ((int)OSKAR_DOUBLE,
                oskar_mem_type(oskar_vis_baseline_ww_metres(vis)));
        ASSERT_EQ(0, oskar_vis_num_channels(vis));
        ASSERT_EQ(0, oskar_vis_num_times(vis));
        ASSERT_EQ(0, oskar_vis_num_baselines(vis));
        ASSERT_EQ(0, oskar_vis_num_stations(vis));
        ASSERT_EQ(0, (int)oskar_mem_length(oskar_vis_amplitude(vis)));
        ASSERT_EQ(0, (int)oskar_mem_length(oskar_vis_baseline_uu_metres(vis)));
        ASSERT_EQ(0, (int)oskar_mem_length(oskar_vis_baseline_vv_metres(vis)));
        ASSERT_EQ(0, (int)oskar_mem_length(oskar_vis_baseline_ww_metres(vis)));
        ASSERT_EQ(0, (int)oskar_mem_length(oskar_vis_baseline_uu_metres(vis)));
        ASSERT_EQ(0, (int)oskar_mem_length(oskar_vis_baseline_vv_metres(vis)));
        ASSERT_EQ(0, (int)oskar_mem_length(oskar_vis_baseline_ww_metres(vis)));
        oskar_vis_free(vis, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }
    {
        int status = 0;

        // Construct visibility data on the GPU and check dimensions.
        oskar_Vis* vis;
        vis = oskar_vis_create(OSKAR_SINGLE_COMPLEX_MATRIX,
                OSKAR_CPU, num_channels, num_times, num_stations,
                &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ((int)OSKAR_CPU, oskar_vis_location(vis));

        ASSERT_EQ((int)OSKAR_SINGLE_COMPLEX_MATRIX,
                oskar_mem_type(oskar_vis_amplitude(vis)));
        ASSERT_EQ((int)OSKAR_SINGLE,
                oskar_mem_type(oskar_vis_baseline_uu_metres(vis)));
        ASSERT_EQ((int)OSKAR_SINGLE,
                oskar_mem_type(oskar_vis_baseline_vv_metres(vis)));
        ASSERT_EQ((int)OSKAR_SINGLE,
                oskar_mem_type(oskar_vis_baseline_ww_metres(vis)));
        ASSERT_EQ(num_channels, oskar_vis_num_channels(vis));
        ASSERT_EQ(num_times, oskar_vis_num_times(vis));
        ASSERT_EQ(num_baselines, oskar_vis_num_baselines(vis));
        ASSERT_EQ(num_stations, oskar_vis_num_stations(vis));
        ASSERT_EQ(num_amps, oskar_vis_num_channels(vis) *
                oskar_vis_num_times(vis) * oskar_vis_num_baselines(vis));
        ASSERT_EQ(num_amps, (int)oskar_mem_length(oskar_vis_amplitude(vis)));
        ASSERT_EQ(num_coords,
                (int)oskar_mem_length(oskar_vis_baseline_uu_metres(vis)));
        ASSERT_EQ(num_coords,
                (int)oskar_mem_length(oskar_vis_baseline_vv_metres(vis)));
        ASSERT_EQ(num_coords,
                (int)oskar_mem_length(oskar_vis_baseline_ww_metres(vis)));
        oskar_vis_free(vis, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }
}


TEST(Visibilities, read_write)
{
    int status = 0;
    int num_channels     = 10;
    int num_times        = 77;
    int num_stations     = 20;
    double start_freq    = 200.0e6;
    double freq_inc      = 10.0e6;
    double time_start_mjd_utc = 10.0;
    double time_inc_seconds   = 1.5;
    int precision        = OSKAR_DOUBLE;
    int amp_type         = precision | OSKAR_COMPLEX;
    const char* filename = "vis_temp.dat";
    oskar_Vis *vis1, *vis2;

    // Create visibilities on the CPU, fill in some data and write to file.
    {
        vis1 = oskar_vis_create(amp_type, OSKAR_CPU,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_vis_set_freq_start_hz(vis1, start_freq);
        oskar_vis_set_freq_inc_hz(vis1, freq_inc);
        oskar_vis_set_time_start_mjd_utc(vis1, time_start_mjd_utc);
        oskar_vis_set_time_inc_sec(vis1, time_inc_seconds);
        const char* name = "dummy";
        oskar_mem_append_raw(oskar_vis_telescope_path(vis1), name,
                OSKAR_CHAR, OSKAR_CPU, 1 + strlen(name), &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        double2* amp = oskar_mem_double2(oskar_vis_amplitude(vis1), &status);
        double* uu = oskar_mem_double(oskar_vis_baseline_uu_metres(vis1), &status);
        double* vv = oskar_mem_double(oskar_vis_baseline_vv_metres(vis1), &status);
        double* ww = oskar_mem_double(oskar_vis_baseline_ww_metres(vis1), &status);

        for (int i = 0, c = 0; c < oskar_vis_num_channels(vis1); ++c)
        {
            for (int t = 0; t < oskar_vis_num_times(vis1); ++t)
            {
                for (int b = 0; b < oskar_vis_num_baselines(vis1); ++b, ++i)
                {
                    amp[i].x = (double)i + 1.123;
                    amp[i].y = (double)i - 0.456;
                }
            }
        }
        for (int i = 0, t = 0; t < oskar_vis_num_times(vis1); ++t)
        {
            for (int b = 0; b < oskar_vis_num_baselines(vis1); ++b, ++i)
            {
                uu[i] = (double)t + 0.1;
                vv[i] = (double)b + 0.2;
                ww[i] = (double)i + 0.3;
            }
        }
        oskar_vis_write(vis1, NULL, filename, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    // Load the visibility structure from file.
    {
        oskar_Binary* h = oskar_binary_create(filename, 'r', &status);
        vis2 = oskar_vis_read(h, &status);
        oskar_binary_free(h);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(amp_type, oskar_mem_type(oskar_vis_amplitude(vis2)));
        ASSERT_EQ(precision, oskar_mem_type(oskar_vis_baseline_uu_metres(vis2)));
        ASSERT_EQ(precision, oskar_mem_type(oskar_vis_baseline_vv_metres(vis2)));
        ASSERT_EQ(precision, oskar_mem_type(oskar_vis_baseline_ww_metres(vis2)));
        ASSERT_EQ((int)OSKAR_CPU, oskar_vis_location(vis2));
        ASSERT_EQ(num_channels, oskar_vis_num_channels(vis2));
        ASSERT_EQ(num_stations * (num_stations - 1) / 2, oskar_vis_num_baselines(vis2));
        ASSERT_EQ(num_times, oskar_vis_num_times(vis2));

        ASSERT_EQ(start_freq, oskar_vis_freq_start_hz(vis2));
        ASSERT_EQ(freq_inc, oskar_vis_freq_inc_hz(vis2));
        ASSERT_EQ(time_start_mjd_utc, oskar_vis_time_start_mjd_utc(vis2));
        ASSERT_EQ(time_inc_seconds, oskar_vis_time_inc_sec(vis2));
        double2* amp = oskar_mem_double2(oskar_vis_amplitude(vis2), &status);
        double* uu = oskar_mem_double(oskar_vis_baseline_uu_metres(vis2), &status);
        double* vv = oskar_mem_double(oskar_vis_baseline_vv_metres(vis2), &status);
        double* ww = oskar_mem_double(oskar_vis_baseline_ww_metres(vis2), &status);

        // Check the data loaded correctly.
        for (int i = 0, c = 0; c < oskar_vis_num_channels(vis2); ++c)
        {
            for (int t = 0; t < oskar_vis_num_times(vis2); ++t)
            {
                for (int b = 0; b < oskar_vis_num_baselines(vis2); ++b, ++i)
                {
                    ASSERT_FLOAT_EQ((double)i + 1.123, amp[i].x);
                    ASSERT_FLOAT_EQ((double)i - 0.456, amp[i].y);
                }
            }
        }
        for (int i = 0, t = 0; t < oskar_vis_num_times(vis2); ++t)
        {
            for (int b = 0; b < oskar_vis_num_baselines(vis2); ++b, ++i)
            {
                ASSERT_FLOAT_EQ((double)t + 0.10, uu[i]);
                ASSERT_FLOAT_EQ((double)b + 0.20, vv[i]);
                ASSERT_FLOAT_EQ((double)i + 0.30, ww[i]);
            }
        }
    }

    // Free memory.
    oskar_vis_free(vis1, &status);
    oskar_vis_free(vis2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Delete temporary file.
    remove(filename);
}
