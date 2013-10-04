/*
 * Copyright (c) 2012-2013, The University of Oxford
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

#include "imaging/oskar_SettingsImage.h"
#include "imaging/oskar_make_image.h"
#include "imaging/oskar_image_write.h"
#include "imaging/oskar_image_resize.h"
#include "imaging/oskar_evaluate_image_lm_grid.h"
#include "imaging/oskar_image_evaluate_ranges.h"

#include <oskar_get_error_string.h>
#include <oskar_vis.h>
#include <oskar_mem_binary_file_write.h>
#include <oskar_mem_binary_stream_write.h>

#ifndef OSKAR_NO_FITS
#include "fits/oskar_fits_image_write.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#define C_0 299792458.0

TEST(make_image, test)
{
    int amp_type      = OSKAR_DOUBLE_COMPLEX_MATRIX;
    int location      = OSKAR_LOCATION_CPU;
    int num_channels  = 2;
    int num_times     = 2;
    int num_stations  = 2;
    int error = 0;

    double freq       = C_0;
    //double lambda     = C_0 / freq;

    oskar_Vis* vis = oskar_vis_create(amp_type, location, num_channels,
            num_times, num_stations, &error);
    double* uu_ = oskar_mem_double(oskar_vis_baseline_uu_metres(vis), &error);
    double* vv_ = oskar_mem_double(oskar_vis_baseline_vv_metres(vis), &error);
    double* ww_ = oskar_mem_double(oskar_vis_baseline_ww_metres(vis), &error);
    double4c* amp_ = oskar_mem_double4c(oskar_vis_amplitude(vis), &error);
    oskar_vis_set_freq_start_hz(vis, freq);
    oskar_vis_set_freq_inc_hz(vis, 10.0e6);

    // time 0, baseline 0
    uu_[0] =  100.0;
    vv_[0] =  0.0;
    ww_[0] =  0.0;
    // time 1, baseline 0
    uu_[1] =  100.0;
    vv_[1] =  0.0;
    ww_[1] =  0.0;

    // channel 0, time 0, baseline 0
    int num_baselines = oskar_vis_num_baselines(vis);
    for (int i = 0, c = 0; c < num_channels; ++c)
    {
        for (int t = 0; t < num_times; ++t)
        {
            for (int b = 0; b < num_baselines; ++b, ++i)
            {
                amp_[i].a.x = 0.0; amp_[i].a.y = 0.0;
                amp_[i].b.x = 0.0; amp_[i].b.y = 4.0;
                amp_[i].c.x = 0.0; amp_[i].c.y = 2.0;
                amp_[i].d.x = 0.0; amp_[i].d.y = 0.0;
            }
        }
    }

    oskar_SettingsImage settings;
    settings.fov_deg = 2.0;
    settings.size    = 256;
    settings.channel_snapshots = OSKAR_TRUE;
    settings.channel_range[0] = 0;
    settings.channel_range[1] = -1;
    settings.time_snapshots = OSKAR_FALSE;
    settings.time_range[0] = 0;
    settings.time_range[1] = -1;
    settings.image_type = OSKAR_IMAGE_TYPE_STOKES;
    settings.direction_type = OSKAR_IMAGE_DIRECTION_OBSERVATION;
    settings.transform_type = OSKAR_IMAGE_DFT_2D;

    oskar_Image image(OSKAR_DOUBLE);
    error = oskar_make_image(&image, NULL, vis, &settings);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

    int idx = 0;
    const char* image_file = "temp_test_image.img";
    oskar_image_write(&image, NULL, image_file, idx, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

#ifndef OSKAR_NO_FITS
    const char* fits_file = "temp_test_image.fits";
    oskar_fits_image_write(&image, NULL, fits_file, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);
#endif

    const char* vis_file = "temp_test_make_image.dat";
    oskar_mem_binary_file_write_ext(oskar_vis_baseline_uu_metres(vis), vis_file,
            "mem", "uu_metres", 0, 0, &error);
    oskar_mem_binary_file_write_ext(oskar_vis_baseline_vv_metres(vis), vis_file,
            "mem", "vv_metres", 0, 0, &error);
    oskar_mem_binary_file_write_ext(oskar_vis_amplitude(vis), vis_file,
            "mem", "vis_amp", 0, 0, &error);
    oskar_mem_binary_file_write_ext(&image.data, vis_file,
            "mem", "image", 0, 0, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

    oskar_vis_free(vis, &error);
    remove(fits_file);
    remove(image_file);
    remove(vis_file);
}

TEST(make_image, image_lm_grid)
{
    // Fill image with lm grid.
    int error = 0;
    int type = OSKAR_DOUBLE;
    int location = OSKAR_LOCATION_CPU;
    int size = 256;
    int num_pixels = size * size;
    double fov = 2.0 * M_PI/180.0;
    oskar_Mem l(type, location, num_pixels);
    oskar_Mem m(type, location, num_pixels);
    oskar_evaluate_image_lm_grid_d(size, size, fov, fov, (double*)l.data,
            (double*)m.data);

    oskar_Image im(OSKAR_DOUBLE);
    oskar_image_resize(&im, size, size, 1, 1, 2, &error);

    memcpy(im.data.data, l.data, num_pixels * sizeof(double));
    memcpy((double*)im.data.data + num_pixels, m.data, num_pixels * sizeof(double));

#ifndef OSKAR_NO_FITS
    const char* fits_file = "test_lm_grid.fits";
    oskar_fits_image_write(&im, NULL, fits_file, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);
#endif
    const char* image_file = "test_lm_grid.img";
    oskar_image_write(&im, NULL, image_file, 0, &error);
    ASSERT_EQ(0, error) << oskar_get_error_string(error);

    remove(fits_file);
    remove(image_file);
}

TEST(make_image, image_range)
{
    int range[2];

    // Use case: snapshots, 0->2, 5 vis times
    // Expect: no fail, image range: 0->2
    {
        int num_vis_times = 5;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {0, 2};

        int err = oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(2, range[1]);
    }

    // Use case: synth, 0->2, 5 vis times
    // Expect: no fail, image range: 0->0
    {
        int num_vis_times = 5;
        int snapshots = OSKAR_FALSE;
        int settings_range[2] = {0, 2};

        int err = oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(0, range[1]);
    }

    // Use case: snapshots, 0->2, 3 vis times
    // Expect: no fail, image range: 0->2
    {
        int num_vis_times = 3;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {0, 2};

        int err = oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(2, range[1]);
    }

    // Use case: snapshots, 0->2, 2 vis times
    // Expect: fail
    {
        int num_vis_times = 2;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {0, 2};

        int err = oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times);
        ASSERT_EQ(OSKAR_ERR_INVALID_RANGE, err) << oskar_get_error_string(err);
    }

    // Use case: snapshots, 3->5, 6 vis times
    // Expect: no fail, range 0->2
    {
        int num_vis_times = 6;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {3, 5};

        int err = oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(2, range[1]);
    }

    // Use case: synth, 3->5, 6 vis times
    // Expect: no fail, range 0->0
    {
        int num_vis_times = 6;
        int snapshots = OSKAR_FALSE;
        int settings_range[2] = {3, 5};

        int err = oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(0, range[1]);
    }

    // Use case: synth, -1->5, 6 vis times
    // Expect: no fail, range 0->0
    {
        int num_vis_times = 6;
        int snapshots = OSKAR_FALSE;
        int settings_range[2] = {-1, 5};

        int err = oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(0, range[1]);
    }

    // Use case: synth, -1->-1, 6 vis times
    // Expect: no fail, range 0->0
    {
        int num_vis_times = 6;
        int snapshots = OSKAR_FALSE;
        int settings_range[2] = {-1, -1};

        int err = oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(0, range[1]);
    }

    // Use case: snapshots, -1->-1, 6 vis times
    // Expect: no fail, range 0->5
    {
        int num_vis_times = 6;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {-1, -1};

        int err = oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(5, range[1]);
    }

    // Use case: snapshots, -1->3, 6 vis times
    // Expect: no fail, range 0->3
    {
        int num_vis_times = 6;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {-1, 3};

        int err = oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(3, range[1]);
    }

    // Use case: snapshots, 1->-1, 6 vis times
    // Expect: no fail, range 0->4
    {
        int num_vis_times = 6;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {1, -1};

        int err = oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(4, range[1]);
    }

    // Use case: snapshots, 5->5, 6 vis times
    // Expect: no fail, range 0->0
    {
        int num_vis_times = 6;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {5, 5};

        int err = oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(0, range[1]);
    }

    // Use case: snapshots, 5->2, 10 vis times
    // Expect: fail
    {
        int num_vis_times = 10;
        int snapshots = OSKAR_TRUE;
        int settings_range[2] = {5, 2};

        int err = oskar_evaluate_image_range(range, snapshots, settings_range,
                num_vis_times);
        ASSERT_EQ(OSKAR_ERR_INVALID_RANGE, err) << oskar_get_error_string(err);
    }
}

TEST(make_image, data_range)
{
    int range[2];

    // Use case: 0->2, 6 vis times
    // Expect: no fail, range 0->2
    {
        int num_vis_times = 6;
        int settings_range[2] = {0, 2};
        int err = oskar_evaluate_image_data_range(range, settings_range,
                num_vis_times);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(2, range[1]);
    }

    // Use case: 2->5, 6 vis times
    // Expect: no fail, range 2->5
    {
        int num_vis_times = 6;
        int settings_range[2] = {2, 5};
        int err = oskar_evaluate_image_data_range(range, settings_range,
                num_vis_times);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(2, range[0]);
        ASSERT_EQ(5, range[1]);
    }

    // Use case: 2->-1, 6 vis times
    // Expect: no fail, range 2->5
    {
        int num_vis_times = 6;
        int settings_range[2] = {2, -1};
        int err = oskar_evaluate_image_data_range(range, settings_range,
                num_vis_times);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(2, range[0]);
        ASSERT_EQ(5, range[1]);
    }

    // Use case: -1->4, 6 vis times
    // Expect: no fail, range 0->4
    {
        int num_vis_times = 6;
        int settings_range[2] = {-1, 4};
        int err = oskar_evaluate_image_data_range(range, settings_range,
                num_vis_times);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(4, range[1]);
    }

    // Use case: -1->-1, 6 vis times
    // Expect: no fail, range 0->5
    {
        int num_vis_times = 6;
        int settings_range[2] = {-1, -1};
        int err = oskar_evaluate_image_data_range(range, settings_range,
                num_vis_times);
        ASSERT_EQ(0, err) << oskar_get_error_string(err);

        ASSERT_EQ(0, range[0]);
        ASSERT_EQ(5, range[1]);
    }

    // Use case: -1->5, 3 vis times
    // Expect: fail
    {
        int num_vis_times = 3;
        int settings_range[2] = {-1, 5};
        int err = oskar_evaluate_image_data_range(range, settings_range,
                num_vis_times);
        ASSERT_EQ(OSKAR_ERR_INVALID_RANGE, err) << oskar_get_error_string(err);
    }

    // Use case: 5->-1, 3 vis times
    // Expect: fail
    {
        int num_vis_times = 3;
        int settings_range[2] = {5, -1};
        int err = oskar_evaluate_image_data_range(range, settings_range,
                num_vis_times);
        ASSERT_EQ(OSKAR_ERR_INVALID_RANGE, err) << oskar_get_error_string(err);
    }

    // Use case: 5->2, 10 vis times
    // Expect: fail
    {
        int num_vis_times = 10;
        int settings_range[2] = {5, 2};
        int err = oskar_evaluate_image_data_range(range, settings_range,
                num_vis_times);
        ASSERT_EQ(OSKAR_ERR_INVALID_RANGE, err) << oskar_get_error_string(err);
    }
}

