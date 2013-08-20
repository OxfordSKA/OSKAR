/*
 * Copyright (c) 2011-2013, The University of Oxford
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

#include <oskar_vector_types.h>
#include <oskar_visibilities_init.h>
#include <oskar_visibilities_init_copy.h>
#include <oskar_visibilities_free.h>
#include <oskar_visibilities_get_channel_amps.h>
#include <oskar_visibilities_location.h>
#include <oskar_visibilities_read.h>
#include <oskar_visibilities_resize.h>
#include <oskar_visibilities_write.h>
#include <oskar_get_error_string.h>
#include <oskar_mem_append_raw.h>
#include <oskar_mem_element_size.h>
#include <oskar_mem_get_pointer.h>
#include <oskar_mem_evaluate_relative_error.h>

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
        oskar_Visibilities vis;
        oskar_visibilities_init(&vis, OSKAR_SINGLE, OSKAR_LOCATION_CPU,
                num_channels, num_times, num_stations, &status);
        EXPECT_EQ(OSKAR_ERR_BAD_DATA_TYPE, status);
        status = 0;
        oskar_visibilities_free(&vis, &status);
        oskar_visibilities_init(&vis, OSKAR_DOUBLE, OSKAR_LOCATION_CPU,
                num_channels, num_times, num_stations, &status);
        EXPECT_EQ(OSKAR_ERR_BAD_DATA_TYPE, status);
        status = 0;
        oskar_visibilities_free(&vis, &status);
    }

    // Don't expect an error for complex types.
    {
        int location, status = 0;
        oskar_Visibilities vis;
        location = OSKAR_LOCATION_CPU;
        oskar_visibilities_init(&vis, OSKAR_DOUBLE_COMPLEX, location,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_visibilities_free(&vis, &status);
        oskar_visibilities_init(&vis, OSKAR_SINGLE_COMPLEX, location,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_visibilities_free(&vis, &status);
        oskar_visibilities_init(&vis, OSKAR_DOUBLE_COMPLEX_MATRIX, location,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_visibilities_free(&vis, &status);
        oskar_visibilities_init(&vis, OSKAR_SINGLE_COMPLEX_MATRIX, location,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_visibilities_free(&vis, &status);
        location = OSKAR_LOCATION_GPU;
        oskar_visibilities_init(&vis, OSKAR_DOUBLE_COMPLEX, location,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_visibilities_free(&vis, &status);
        oskar_visibilities_init(&vis, OSKAR_SINGLE_COMPLEX, location,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_visibilities_free(&vis, &status);
        oskar_visibilities_init(&vis, OSKAR_DOUBLE_COMPLEX_MATRIX, location,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_visibilities_free(&vis, &status);
        oskar_visibilities_init(&vis, OSKAR_SINGLE_COMPLEX_MATRIX, location,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_visibilities_free(&vis, &status);
    }
    {
        int status = 0;

        // Construct visibility data on the CPU and check dimensions.
        oskar_Visibilities vis;
        oskar_visibilities_init(&vis, OSKAR_SINGLE_COMPLEX_MATRIX,
                OSKAR_LOCATION_CPU, num_channels, num_times, num_stations,
                &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ((int)OSKAR_LOCATION_CPU, oskar_visibilities_location(&vis));

        ASSERT_EQ((int)OSKAR_SINGLE_COMPLEX_MATRIX, vis.amplitude.type);
        ASSERT_EQ((int)OSKAR_SINGLE, vis.uu_metres.type);
        ASSERT_EQ((int)OSKAR_SINGLE, vis.vv_metres.type);
        ASSERT_EQ((int)OSKAR_SINGLE, vis.ww_metres.type);
        ASSERT_EQ(num_channels, vis.num_channels);
        ASSERT_EQ(num_times, vis.num_times);
        ASSERT_EQ(num_baselines, vis.num_baselines);
        ASSERT_EQ(num_stations, vis.num_stations);
        ASSERT_EQ(num_amps, vis.num_channels * vis.num_times * vis.num_baselines);
        ASSERT_EQ(num_amps, vis.amplitude.num_elements);
        ASSERT_EQ(num_coords, vis.uu_metres.num_elements);
        ASSERT_EQ(num_coords, vis.vv_metres.num_elements);
        ASSERT_EQ(num_coords, vis.ww_metres.num_elements);
        oskar_visibilities_free(&vis, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        oskar_visibilities_init(&vis, OSKAR_DOUBLE_COMPLEX_MATRIX,
                OSKAR_LOCATION_CPU, 0, 0, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ((int)OSKAR_LOCATION_CPU, oskar_visibilities_location(&vis));
        ASSERT_EQ((int)OSKAR_DOUBLE_COMPLEX_MATRIX, vis.amplitude.type);
        ASSERT_EQ((int)OSKAR_DOUBLE, vis.uu_metres.type);
        ASSERT_EQ((int)OSKAR_DOUBLE, vis.vv_metres.type);
        ASSERT_EQ((int)OSKAR_DOUBLE, vis.ww_metres.type);
        ASSERT_EQ(0, vis.num_channels);
        ASSERT_EQ(0, vis.num_times);
        ASSERT_EQ(0, vis.num_baselines);
        ASSERT_EQ(0, vis.num_stations);
        ASSERT_EQ(0, vis.amplitude.num_elements);
        ASSERT_EQ(0, vis.uu_metres.num_elements);
        ASSERT_EQ(0, vis.vv_metres.num_elements);
        ASSERT_EQ(0, vis.ww_metres.num_elements);
        ASSERT_EQ(NULL, vis.uu_metres.data);
        ASSERT_EQ(NULL, vis.vv_metres.data);
        ASSERT_EQ(NULL, vis.ww_metres.data);
        ASSERT_EQ(NULL, vis.amplitude.data);
        ASSERT_EQ(0, vis.uu_metres.num_elements);
        ASSERT_EQ(0, vis.vv_metres.num_elements);
        ASSERT_EQ(0, vis.ww_metres.num_elements);
        oskar_visibilities_free(&vis, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }
    {
        int status = 0;

        // Construct visibility data on the GPU and check dimensions.
        oskar_Visibilities vis;
        oskar_visibilities_init(&vis, OSKAR_SINGLE_COMPLEX_MATRIX,
                OSKAR_LOCATION_CPU, num_channels, num_times, num_stations,
                &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ((int)OSKAR_LOCATION_CPU, oskar_visibilities_location(&vis));

        ASSERT_EQ((int)OSKAR_SINGLE_COMPLEX_MATRIX, vis.amplitude.type);
        ASSERT_EQ((int)OSKAR_SINGLE, vis.uu_metres.type);
        ASSERT_EQ((int)OSKAR_SINGLE, vis.vv_metres.type);
        ASSERT_EQ((int)OSKAR_SINGLE, vis.ww_metres.type);
        ASSERT_EQ(num_channels, vis.num_channels);
        ASSERT_EQ(num_times, vis.num_times);
        ASSERT_EQ(num_baselines, vis.num_baselines);
        ASSERT_EQ(num_stations, vis.num_stations);
        ASSERT_EQ(num_amps, vis.num_channels * vis.num_times * vis.num_baselines);
        ASSERT_EQ(num_amps, vis.amplitude.num_elements);
        ASSERT_EQ(num_coords, vis.uu_metres.num_elements);
        ASSERT_EQ(num_coords, vis.vv_metres.num_elements);
        ASSERT_EQ(num_coords, vis.ww_metres.num_elements);
        oskar_visibilities_free(&vis, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }
}


TEST(Visibilities, copy)
{
    oskar_Visibilities vis1, vis2, vis3;
    double min_rel_error = 0.0, max_rel_error = 0.0;
    double avg_rel_error = 0.0, std_rel_error = 0.0;
    int num_channels = 2, num_times = 3, num_stations = 2, status = 0;

    // Create and fill a visibility structure on the CPU.
    oskar_visibilities_init(&vis1, OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_CPU,
            num_channels, num_times, num_stations, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0, c = 0; c < vis1.num_channels; ++c)
    {
        for (int t = 0; t < vis1.num_times; ++t)
        {
            for (int b = 0; b < vis1.num_baselines; ++b, ++i)
            {
                ((float2*)vis1.amplitude.data)[i].x  = (float)i + 1.123f;
                ((float2*)vis1.amplitude.data)[i].y  = (float)i - 0.456f;
            }
        }
    }
    for (int i = 0, t = 0; t < vis1.num_times; ++t)
    {
        for (int b = 0; b < vis1.num_baselines; ++b, ++i)
        {
            ((float*)vis1.uu_metres.data)[i] = (float)b + 0.1f;
            ((float*)vis1.vv_metres.data)[i] = (float)t + 0.2f;
            ((float*)vis1.ww_metres.data)[i] = (float)i + 0.3f;
        }
    }

    // Copy to GPU.
    oskar_visibilities_init_copy(&vis2, &vis1, OSKAR_LOCATION_GPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ((int)OSKAR_LOCATION_GPU, oskar_visibilities_location(&vis2));
    ASSERT_EQ(vis1.num_channels, vis2.num_channels);
    ASSERT_EQ(vis1.num_times, vis2.num_times);
    ASSERT_EQ(vis1.num_stations, vis2.num_stations);
    ASSERT_EQ(vis1.num_baselines, vis2.num_baselines);

    // Copy back to CPU and check values.
    oskar_visibilities_init_copy(&vis3, &vis2, OSKAR_LOCATION_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ((int)OSKAR_LOCATION_CPU, oskar_visibilities_location(&vis3));
    ASSERT_EQ(vis1.num_channels, vis3.num_channels);
    ASSERT_EQ(vis1.num_times, vis3.num_times);
    ASSERT_EQ(vis1.num_stations, vis3.num_stations);
    ASSERT_EQ(vis1.num_baselines, vis3.num_baselines);
    oskar_mem_evaluate_relative_error(&vis3.amplitude, &vis1.amplitude,
            &min_rel_error, &max_rel_error, &avg_rel_error, &std_rel_error,
            &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_LT(max_rel_error, 1e-10);
    ASSERT_LT(avg_rel_error, 1e-10);
    oskar_mem_evaluate_relative_error(&vis3.uu_metres, &vis1.uu_metres,
            &min_rel_error, &max_rel_error, &avg_rel_error, &std_rel_error,
            &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_LT(max_rel_error, 1e-10);
    ASSERT_LT(avg_rel_error, 1e-10);
    oskar_mem_evaluate_relative_error(&vis3.vv_metres, &vis1.vv_metres,
            &min_rel_error, &max_rel_error, &avg_rel_error, &std_rel_error,
            &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_LT(max_rel_error, 1e-10);
    ASSERT_LT(avg_rel_error, 1e-10);
    oskar_mem_evaluate_relative_error(&vis3.ww_metres, &vis1.ww_metres,
            &min_rel_error, &max_rel_error, &avg_rel_error, &std_rel_error,
            &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_LT(max_rel_error, 1e-10);
    ASSERT_LT(avg_rel_error, 1e-10);

    // Free memory.
    oskar_visibilities_free(&vis1, &status);
    oskar_visibilities_free(&vis2, &status);
    oskar_visibilities_free(&vis3, &status);
}


TEST(Visibilities, resize)
{
    int status = 0;
    {
        oskar_Visibilities vis;
        oskar_visibilities_init(&vis, OSKAR_DOUBLE_COMPLEX,
                OSKAR_LOCATION_CPU, 0, 0, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ((int)OSKAR_LOCATION_CPU, oskar_visibilities_location(&vis));
        ASSERT_EQ(0, vis.num_channels);
        ASSERT_EQ(0, vis.num_times);
        ASSERT_EQ(0, vis.num_stations);
        ASSERT_EQ(0, vis.num_baselines);
        oskar_visibilities_resize(&vis, 5, 10, 2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(5, vis.num_channels);
        ASSERT_EQ(10, vis.num_times);
        ASSERT_EQ(2, vis.num_stations);
        ASSERT_EQ(1, vis.num_baselines);
        oskar_visibilities_free(&vis, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    {
        oskar_Visibilities vis;
        oskar_visibilities_init(&vis, OSKAR_SINGLE_COMPLEX,
                OSKAR_LOCATION_GPU, 0, 0, 0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ((int)OSKAR_LOCATION_GPU, oskar_visibilities_location(&vis));
        ASSERT_EQ(0, vis.num_channels);
        ASSERT_EQ(0, vis.num_times);
        ASSERT_EQ(0, vis.num_stations);
        ASSERT_EQ(0, vis.num_baselines);
        oskar_visibilities_resize(&vis, 5, 10, 2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(5, vis.num_channels);
        ASSERT_EQ(10, vis.num_times);
        ASSERT_EQ(2, vis.num_stations);
        ASSERT_EQ(1, vis.num_baselines);
        oskar_visibilities_free(&vis, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }
}


TEST(Visibilities, get_channel_amps)
{
    int amp_type     = OSKAR_SINGLE_COMPLEX;
    int location     = OSKAR_LOCATION_CPU;
    int num_channels = 5;
    int num_times    = 4;
    int num_stations = 3;
    int status = 0;

    oskar_Visibilities vis;
    oskar_visibilities_init(&vis, amp_type, location,
            num_channels, num_times, num_stations, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    for (int i = 0, c = 0; c < vis.num_channels; ++c)
    {
        for (int t = 0; t < vis.num_times; ++t)
        {
            for (int b = 0; b < vis.num_baselines; ++b, ++i)
            {
                ((float2*)vis.amplitude.data)[i].x  = (float)c + 1.123;
                ((float2*)vis.amplitude.data)[i].y  = ((float)(i + c) - 0.456);
            }
        }
    }

    for (int i = 0, c = 0; c < vis.num_channels; ++c)
    {
        oskar_Mem vis_amps;
        oskar_visibilities_get_channel_amps(&vis_amps, &vis, c, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(num_times * vis.num_baselines, vis_amps.num_elements);
        ASSERT_EQ(amp_type, vis_amps.type);
        ASSERT_EQ(location, vis_amps.location);
        ASSERT_EQ(0, vis_amps.owner);

        for (int t = 0; t < vis.num_times; ++t)
        {
            for (int b = 0; b < vis.num_baselines; ++b, ++i)
            {
                ASSERT_FLOAT_EQ((float)c + 1.123,
                        ((float2*)vis_amps.data)[t * vis.num_baselines + b].x);
                ASSERT_FLOAT_EQ(((float)(i + c) - 0.456),
                        ((float2*)vis_amps.data)[t * vis.num_baselines + b].y);
            }
        }
    }
    oskar_visibilities_free(&vis, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}


TEST(Visibilities, read_write)
{
    int status = 0;
    int num_channels     = 10;
    int num_times        = 10;
    int num_stations     = 20;
    double start_freq    = 200.0e6;
    double freq_inc      = 10.0e6;
    double time_start_mjd_utc = 10.0;
    double time_inc_seconds   = 1.5;
    int precision        = OSKAR_SINGLE;
    int amp_type         = precision | OSKAR_COMPLEX;
    const char* filename = "vis_temp.dat";
    oskar_Visibilities vis1, vis2;

    // Create visibilities on the CPU, fill in some data and write to file.
    {
        oskar_visibilities_init(&vis1, amp_type, OSKAR_LOCATION_CPU,
                num_channels, num_times, num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        vis1.freq_start_hz      = 200e6;
        vis1.freq_inc_hz        = 10e6;
        vis1.time_start_mjd_utc = 10.0;
        vis1.time_inc_seconds   = 1.5;
        const char* name = "dummy";
        oskar_mem_append_raw(&vis1.telescope_path, name,
                OSKAR_CHAR, OSKAR_LOCATION_CPU, 1 + strlen(name), &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        for (int i = 0, c = 0; c < vis1.num_channels; ++c)
        {
            for (int t = 0; t < vis1.num_times; ++t)
            {
                for (int b = 0; b < vis1.num_baselines; ++b, ++i)
                {
                    ((float2*)vis1.amplitude.data)[i].x  = (float)i + 1.123f;
                    ((float2*)vis1.amplitude.data)[i].y  = (float)i - 0.456f;
                }
            }
        }
        for (int i = 0, t = 0; t < vis1.num_times; ++t)
        {
            for (int b = 0; b < vis1.num_baselines; ++b, ++i)
            {
                ((float*)vis1.uu_metres.data)[i]     = (float)t + 0.1f;
                ((float*)vis1.vv_metres.data)[i]     = (float)b + 0.2f;
                ((float*)vis1.ww_metres.data)[i]     = (float)i + 0.3f;
            }
        }
        oskar_visibilities_write(&vis1, NULL, filename, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    // Load the visibility structure from file.
    {
        oskar_visibilities_read(&vis2, filename, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(amp_type, vis2.amplitude.type);
        ASSERT_EQ(precision, vis2.uu_metres.type);
        ASSERT_EQ(precision, vis2.vv_metres.type);
        ASSERT_EQ(precision, vis2.ww_metres.type);
        ASSERT_EQ((int)OSKAR_LOCATION_CPU, oskar_visibilities_location(&vis2));
        ASSERT_EQ(num_channels, vis2.num_channels);
        ASSERT_EQ(num_stations * (num_stations - 1) / 2, vis2.num_baselines);
        ASSERT_EQ(num_times, vis2.num_times);

        ASSERT_EQ(start_freq, vis2.freq_start_hz);
        ASSERT_EQ(freq_inc, vis2.freq_inc_hz);
        ASSERT_EQ(time_start_mjd_utc, vis2.time_start_mjd_utc);
        ASSERT_EQ(time_inc_seconds, vis2.time_inc_seconds);

        // Check the data loaded correctly.
        for (int i = 0, c = 0; c < vis2.num_channels; ++c)
        {
            for (int t = 0; t < vis2.num_times; ++t)
            {
                for (int b = 0; b < vis2.num_baselines; ++b, ++i)
                {
                    ASSERT_FLOAT_EQ((float)i + 1.123f,
                            ((float2*)vis2.amplitude.data)[i].x);
                    ASSERT_FLOAT_EQ((float)i - 0.456f,
                            ((float2*)vis2.amplitude.data)[i].y);
                }
            }
        }
        for (int i = 0, t = 0; t < vis2.num_times; ++t)
        {
            for (int b = 0; b < vis2.num_baselines; ++b, ++i)
            {
                ASSERT_FLOAT_EQ((float)t + 0.10f, ((float*)vis2.uu_metres.data)[i]);
                ASSERT_FLOAT_EQ((float)b + 0.20f, ((float*)vis2.vv_metres.data)[i]);
                ASSERT_FLOAT_EQ((float)i + 0.30f, ((float*)vis2.ww_metres.data)[i]);
            }
        }
    }

    // Free memory.
    oskar_visibilities_free(&vis1, &status);
    oskar_visibilities_free(&vis2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Delete temporary file.
    remove(filename);
}
