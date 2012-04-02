/*
 * Copyright (c) 2011, The University of Oxford
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

#include "imaging/test/Test_make_image.h"

#include "imaging/oskar_Image.h"
#include "imaging/oskar_SettingsImage.h"
#include "imaging/oskar_make_image.h"
#include "imaging/oskar_image_write.h"
#include "imaging/oskar_image_resize.h"
#include "imaging/oskar_evaluate_image_lm_grid.h"

#include "interferometry/oskar_Visibilities.h"
#include "interferometry/oskar_visibilities_init.h"

#include "utility/oskar_vector_types.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_mem_binary_file_write.h"
#include "utility/oskar_mem_binary_stream_write.h"

#ifndef OSKAR_NO_FITS
#include "fits/oskar_fits_image_write.h"
#endif

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>

#define C_0 299792458.0

void Test_make_image::test()
{
    int amp_type      = OSKAR_DOUBLE_COMPLEX_MATRIX;
    int location      = OSKAR_LOCATION_CPU;
    int num_channels  = 2;
    int num_times     = 2;
    int num_baselines = 2;

    double freq       = C_0;
    //double lambda     = C_0 / freq;

    oskar_Visibilities vis;
    oskar_visibilities_init(&vis, amp_type, location, num_channels, num_times,
            num_baselines);
    double* uu_ = (double*)vis.uu_metres.data;
    double* vv_ = (double*)vis.vv_metres.data;
    double* ww_ = (double*)vis.ww_metres.data;
    double4c* amp_ = (double4c*)vis.amplitude.data;
    vis.freq_start_hz = freq;
    vis.freq_inc_hz   = 10.0e6;

    // time 0, baseline 0
    uu_[0] =  1.0;
    vv_[0] =  0.0;
    ww_[0] =  0.0;
    // time 0, baseline 1
    uu_[1] =  0.0;
    vv_[1] =  1.0;
    ww_[1] =  0.0;
    // time 1, baseline 0
    uu_[2] =  1.0;
    vv_[2] =  1.0;
    ww_[2] =  0.0;
    // time 1, baseline 1
    uu_[3] = -1.0;
    vv_[3] =  1.0;
    ww_[3] =  0.0;

    // channel 0, time 0, baseline 0
    for (int i = 0, c = 0; c < num_channels; ++c)
    {
        for (int t = 0; t < num_times; ++t)
        {
            for (int b = 0; b < num_baselines; ++b, ++i)
            {
                amp_[i].a.x = 1.0; amp_[i].a.y = 0.0;
                amp_[i].b.x = 1.0; amp_[i].b.y = 0.0;
                amp_[i].c.x = 1.0; amp_[i].c.y = 0.0;
                amp_[i].d.x = 1.0; amp_[i].d.y = 0.0;
            }
        }
    }

    oskar_SettingsImage settings;
    settings.fov_deg = 2.0;
    settings.size    = 256;
    settings.channel_snapshots = OSKAR_TRUE;
    settings.channel_range[0] = 0;
    settings.channel_range[1] = 1;
    settings.time_snapshots = OSKAR_TRUE;
    settings.time_range[0] = 0;
    settings.time_range[1] = 1;
    settings.image_type = OSKAR_IMAGE_TYPE_POL_LINEAR;
    settings.transform_type = OSKAR_IMAGE_DFT_2D;

    oskar_Image image;
    int err = oskar_make_image(&image, &vis, &settings);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), (int)OSKAR_SUCCESS, err);

    int idx = 0;
    err = oskar_image_write(&image, "temp_test_image.img", idx);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), (int)OSKAR_SUCCESS, err);


    const char* filename = "temp_test_make_image.dat";
    oskar_mem_binary_file_write_ext(&vis.uu_metres, filename,
            "mem", "uu_metres", 0, 0);
    oskar_mem_binary_file_write_ext(&vis.vv_metres, filename,
                "mem", "vv_metres", 0, 0);
    oskar_mem_binary_file_write_ext(&vis.amplitude, filename,
                    "mem", "vis_amp", 0, 0);
    oskar_mem_binary_file_write_ext(&image.data, filename,
                        "mem", "image", 0, 0);
}

void  Test_make_image::image_lm_grid()
{
    // Fill image with lm grid.
    int type = OSKAR_DOUBLE;
    int location = OSKAR_LOCATION_CPU;
    int size = 256;
    int num_pixels = size * size;
    double fov = 2.0 * M_PI/180.0;
    oskar_Mem l(type, location, num_pixels);
    oskar_Mem m(type, location, num_pixels);
    oskar_evaluate_image_lm_grid_d(size, size, fov, fov, (double*)l.data,
            (double*)m.data);

    oskar_Image im;
    oskar_image_resize(&im, size, size, 1, 1, 2);

    memcpy(im.data.data, l.data, num_pixels * sizeof(double));
    memcpy((double*)im.data.data + num_pixels, m.data, num_pixels * sizeof(double));

#ifndef OSKAR_NO_FITS
    oskar_fits_image_write(&im, "test_lm_grid.fits");
#endif
    oskar_image_write(&im, "test_lm_grid.img", 0);
}
