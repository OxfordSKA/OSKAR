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

#include <apps/lib/oskar_vis_write_ms.h>
#include <apps/lib/oskar_remove_dir.h>

#include <oskar_date_time_to_mjd.h>
#include <oskar_vis.h>
#include <oskar_telescope.h>
#include <oskar_get_error_string.h>

#include <cstdio>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

TEST(write_ms, test_write)
{
    int status = 0;
    int num_antennas  = 5;
    int num_channels  = 3;
    int num_times     = 5;
    int num_baselines = num_antennas * (num_antennas - 1) / 2;

    // Create a visibility structure and fill in some data.
    oskar_Vis* vis = oskar_vis_create(OSKAR_DOUBLE_COMPLEX_MATRIX,
            OSKAR_LOCATION_CPU, num_channels, num_times, num_antennas, &status);
    ASSERT_TRUE(vis);
    oskar_vis_set_phase_centre(vis, 160.0, 89.0);
    double4c* v_ = oskar_mem_double4c(oskar_vis_amplitude(vis), &status);
    double *uu, *vv, *ww, *x, *y, *z;
    uu = oskar_mem_double(oskar_vis_baseline_uu_metres(vis), &status);
    vv = oskar_mem_double(oskar_vis_baseline_vv_metres(vis), &status);
    ww = oskar_mem_double(oskar_vis_baseline_ww_metres(vis), &status);
    x = oskar_mem_double(oskar_vis_station_x_metres(vis), &status);
    y = oskar_mem_double(oskar_vis_station_y_metres(vis), &status);
    z = oskar_mem_double(oskar_vis_station_z_metres(vis), &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    for (int i = 0, c = 0; c < num_channels; ++c)
    {
        for (int t = 0; t < num_times; ++t)
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
    oskar_vis_set_freq_start_hz(vis, 222.22e6);
    oskar_vis_set_freq_inc_hz(vis, 11.1e6);
    oskar_vis_set_time_start_mjd_utc(vis,
            oskar_date_time_to_mjd(2011, 11, 17, 0.0));
    oskar_vis_set_time_inc_seconds(vis, 1.0);

    const char* filename = "temp_test_write_ms.ms";

    oskar_vis_write_ms(vis, filename, OSKAR_TRUE, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_remove_dir(filename);
}

