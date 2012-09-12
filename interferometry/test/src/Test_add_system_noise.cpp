/*
 * Copyright (c) 2012, The University of Oxford
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


#include "interferometry/test/Test_add_system_noise.h"
#include "interferometry/oskar_visibilities_add_system_noise.h"
#include "interferometry/oskar_telescope_model_init.h"
#include "interferometry/oskar_visibilities_init.h"
#include "interferometry/oskar_visibilities_write.h"
#include "interferometry/oskar_evaluate_uvw_baseline.h"

#include "station/oskar_station_model_init.h"

#include "utility/oskar_Mem.h"
#include "utility/oskar_mem_realloc.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_Settings.h"
#include "utility/oskar_settings_init.h"

#include "imaging/oskar_make_image.h"
#include "imaging/oskar_image_write.h"

#include "math/oskar_random_gaussian.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <algorithm>
#include <limits>
#include <cfloat>
#include <vector>

void Test_add_system_noise::test_rms()
{
    int err = OSKAR_SUCCESS;
    int type = OSKAR_DOUBLE;
    int location = OSKAR_LOCATION_CPU;
    int seed = 0;

    // Setup some settings
    oskar_Settings settings;
    oskar_settings_init(&settings);

    // Setup the telescope model.
    oskar_TelescopeModel telescope;
    int num_stations = 10;
    int num_noise_values = 2;
    double freq_start = 20.0e6;
    double freq_inc   = 10.0e6;
    double stddev_start = 1.0;
    double stddev_inc = 1.0;
    double r_stddev = 5000.0;
    err = oskar_telescope_model_init(&telescope, type, location, num_stations);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

    for (int i = 0; i < num_stations; ++i)
    {
        if (type == OSKAR_DOUBLE)
        {
            double* x = (double*)telescope.station_x.data;
            double* y = (double*)telescope.station_y.data;
            double* z = (double*)telescope.station_z.data;
            double r1, r2;
            r1 = oskar_random_gaussian(&r2);
            x[i] = r1 * r_stddev;
            y[i] = r2 * r_stddev;
            z[i] = 0.0;
        }
        else
        {
            float* x = (float*)telescope.station_x.data;
            float* y = (float*)telescope.station_y.data;
            float* z = (float*)telescope.station_z.data;
            double r1, r2;
            r1 = oskar_random_gaussian(&r2);
            x[i] = r1 * r_stddev;
            y[i] = r2 * r_stddev;
            z[i] = 0.0;
        }

        oskar_Mem* freqs = &(telescope.station[i].noise.frequency);
        generate_range(freqs, num_noise_values, freq_start, freq_inc);

        oskar_Mem* stddev = &(telescope.station[i].noise.rms);
        generate_range(stddev, num_noise_values, stddev_start, stddev_inc);
    }
    telescope.ra0_rad = 0.0;
    telescope.dec0_rad = 60.0 * (180.0 / M_PI);

    // Setup the visibilities structure.
    oskar_Visibilities vis;
    int num_channels = 1;
    int num_times = 5;
    err = oskar_visibilities_init(&vis, type | OSKAR_COMPLEX | OSKAR_MATRIX,
            location, num_channels, num_times, num_stations);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
    vis.freq_start_hz = freq_start;
    vis.freq_inc_hz = freq_inc;
    vis.time_start_mjd_utc = 56127.0;
    vis.time_inc_seconds = 100.0;
    vis.channel_bandwidth_hz = 0.15e6;
    vis.phase_centre_ra_deg = telescope.ra0_rad * (180.0/M_PI);
    vis.phase_centre_dec_deg = telescope.dec0_rad * (180.0/M_PI);
    vis.num_stations = num_stations;

    err = oskar_visibilities_add_system_noise(&vis, &telescope, seed);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

    // Evaluate baseline coordinates
    settings.obs.ra0_rad = telescope.ra0_rad;
    settings.obs.dec0_rad = telescope.dec0_rad;
    settings.obs.start_frequency_hz = vis.freq_start_hz;
    settings.obs.num_channels = num_channels;
    settings.obs.frequency_inc_hz = vis.freq_inc_hz;
    settings.obs.num_time_steps = num_times;
    settings.obs.start_mjd_utc = vis.time_start_mjd_utc;
    settings.obs.length_seconds = num_times * vis.time_inc_seconds;
    settings.obs.length_days = settings.obs.length_seconds / 86400.0;
    settings.obs.dt_dump_days = vis.time_inc_seconds / 86400.0;

    oskar_Mem work_uvw(type, OSKAR_LOCATION_CPU, 3 * num_stations);
    oskar_evaluate_uvw_baseline(&vis.uu_metres, &vis.vv_metres,
            &vis.ww_metres, telescope.num_stations, &telescope.station_x,
            &telescope.station_y, &telescope.station_z, telescope.ra0_rad,
            telescope.dec0_rad, settings.obs.num_time_steps,
            settings.obs.start_mjd_utc, settings.obs.dt_dump_days,
            &work_uvw, &err);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

//    err = oskar_visibilities_write(&vis, NULL, "temp_test.vis");
//    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

    oskar_Image image;
    settings.image.input_vis_data = NULL;
    settings.image.size = 1024;
    settings.image.fov_deg = 0.75;
    settings.image.image_type = OSKAR_IMAGE_TYPE_POL_XX;
    settings.image.channel_snapshots = OSKAR_TRUE;
    settings.image.channel_range[0] = 0;
    settings.image.channel_range[1] = -1;
    settings.image.time_snapshots = OSKAR_TRUE;
    settings.image.time_range[0] = 0;
    settings.image.time_range[1] = -1;
    settings.image.transform_type = OSKAR_IMAGE_DFT_2D;
    settings.image.direction_type = OSKAR_IMAGE_DIRECTION_OBSERVATION;
    std::string filename = "temp_test_image.img";
    settings.image.oskar_image = (char*)malloc(filename.size() + 1);
    strcpy(settings.image.oskar_image, filename.c_str());
    settings.image.fits_image = NULL;


    try
    {
        err = oskar_make_image(&image, 0, &vis, &(settings.image));
        CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);
    }
    catch (const int& err)
    {
        CPPUNIT_ASSERT_EQUAL((int)OSKAR_SUCCESS, err);
    }

    //    err = oskar_image_write(&image, NULL, settings.image.oskar_image, 0);
    //    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

    //check_image_stats(&image, &telescope);
}

void Test_add_system_noise::generate_range(oskar_Mem* data, int number, double start, double inc)
{
    int err = 0;
    oskar_mem_realloc(data, number, &err);
    CPPUNIT_ASSERT_EQUAL_MESSAGE(oskar_get_error_string(err), 0, err);

    for (int i = 0; i < number; ++i)
    {
        if (data->type == OSKAR_DOUBLE)
            ((double*)data->data)[i] = start + i * inc;
        else
            ((float*)data->data)[i] = start + i * inc;
    }
}

void Test_add_system_noise::check_image_stats(oskar_Image* image,
        oskar_TelescopeModel* tel)
{
    int num_pixels = image->width * image->height;
    int type = image->data.type;
    int num_channels = image->num_channels;
    int num_pols = image->num_pols;
    int num_times = image->num_times;
    std::vector<double> ave_rms(num_channels * num_pols, 0.0);
    std::vector<double> ave_mean(num_channels * num_pols, 0.0);

    for (int slice = 0, c = 0; c < num_channels; ++c)
    {
        for (int t = 0; t < num_times; ++t)
        {
            for (int p = 0; p < num_pols; ++p)
            {
                int offset = slice * num_pixels;
                double im_max = -DBL_MAX;
                double im_min = DBL_MAX;
                double im_sum = 0.0;
                double im_sum_sq = 0.0;
                //printf("(c: %i, t: %i, p: %i) slice = %i, offset = %i\n", c, t, p, slice, offset);

                for (int i = 0; i < num_pixels; ++i)
                {
                    if (type == OSKAR_DOUBLE)
                    {
                        double* im = (double*)image->data.data + offset;
                        im_max = std::max<double>(im_max, im[i]);
                        im_min = std::min<double>(im_min, im[i]);
                        im_sum += im[i];
                        im_sum_sq += im[i]*im[i];
                    }
                    else
                    {
                        float* im = (float*)image->data.data + offset;
                        im_max = std::max<float>(im_max, im[i]);
                        im_min = std::min<float>(im_min, im[i]);
                        im_sum += im[i];
                        im_sum_sq += im[i]*im[i];
                    }
                }
                double im_mean = im_sum / num_pixels;
                double im_rms = sqrt(im_sum_sq / num_pixels);
                //                printf("    min  = %f\n", im_min);
                //                printf("    max  = %f\n", im_max);
                //                printf("    mean = %f\n", im_mean);
                //                printf("    rms  = %f\n", im_rms);
                ave_rms[c * num_pols + p] += im_rms;
                ave_mean[c * num_pols + p] += im_mean;
                slice++;
            }
        }
    }
    oskar_Mem* s = &tel->station[0].noise.rms;
    int num_baselines = (tel->num_stations * tel->num_stations - 1) / 2;

    for (int c = 0; c < num_channels; ++c)
    {
        for (int p = 0; p < num_pols; ++p)
        {
            ave_rms[c * num_pols + p] /= num_times;
            ave_mean[c * num_pols + p] /= num_times;
            double expected = 0.0;
            if (type == OSKAR_DOUBLE)
            {
                expected = ((double*)s->data)[c] / sqrt((double)num_baselines);
            }
            else
            {
                expected = ((float*)s->data)[c] / sqrt((float)num_baselines);
            }
            printf("(c %i, p %i) rms = % -f (%f, %f), mean = % -f\n", c, p,
                    ave_rms[c * num_pols + p], expected,
                    fabs(ave_rms[c * num_pols + p] - expected),
                    ave_mean[c * num_pols + p]);
        }
    }
}

