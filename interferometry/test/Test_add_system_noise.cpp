/*
 * Copyright (c) 2012-2015, The University of Oxford
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

#include <oskar_telescope.h>
#include <oskar_vis.h>
#include <oskar_convert_ecef_to_baseline_uvw.h>

#include <oskar_get_error_string.h>
#include <oskar_settings_init.h>

#include <oskar_make_image.h>
#include <oskar_image.h>

#include <oskar_random_gaussian.h>
#include <oskar_cmath.h>

#include <cstdio>
#include <cstdlib>
#include <string>
#include <cstring>
#include <algorithm>
#include <limits>
#include <cfloat>
#include <vector>

static void generate_range(oskar_Mem* data, int number,
        double start, double inc);
#if 0
static void check_image_stats(oskar_Image* image, oskar_Telescope* tel);
#endif


TEST(add_system_noise, test_rms)
{
    int err = OSKAR_SUCCESS;
    int type = OSKAR_DOUBLE;
    int location = OSKAR_CPU;
    int seed = 0;

    // Set up some settings.
    oskar_Settings settings;
    oskar_settings_init(&settings);

    // Set up the telescope model.
    int num_stations = 10;
    int num_noise_values = 2;
    double freq_start = 20.0e6;
    double freq_inc   = 10.0e6;
    double stddev_start = 1.0;
    double stddev_inc = 1.0;
    double r_stddev = 5000.0;
    double ra0_rad = 0.0;
    double dec0_rad = 60.0 * M_PI / 180.0;
    oskar_Telescope* tel = oskar_telescope_create(type,
            location, num_stations, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);

    oskar_mem_random_gaussian(
            oskar_telescope_station_true_x_offset_ecef_metres(tel),
            seed, 0, 0, 0, r_stddev, &err);
    oskar_mem_random_gaussian(
            oskar_telescope_station_true_y_offset_ecef_metres(tel),
            seed, 1, 0, 0, r_stddev, &err);
    oskar_mem_clear_contents(
            oskar_telescope_station_true_z_offset_ecef_metres(tel), &err);

    for (int i = 0; i < num_stations; ++i)
    {
        oskar_Station* st = oskar_telescope_station(tel, i);
        generate_range(oskar_station_noise_freq_hz(st), num_noise_values,
                freq_start, freq_inc);
        generate_range(oskar_station_noise_rms_jy(st), num_noise_values,
                stddev_start, stddev_inc);
    }

    oskar_telescope_set_phase_centre(tel,
            OSKAR_SPHERICAL_TYPE_EQUATORIAL, ra0_rad, dec0_rad);

    // Setup the visibilities structure.
    oskar_Vis* vis;
    int num_channels = 1;
    int num_times = 5;
    vis = oskar_vis_create(type | OSKAR_COMPLEX | OSKAR_MATRIX,
            location, num_channels, num_times, num_stations, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);
    oskar_vis_set_freq_start_hz(vis, freq_start);
    oskar_vis_set_freq_inc_hz(vis, freq_inc);
    oskar_vis_set_time_start_mjd_utc(vis, 56127.0);
    oskar_vis_set_time_inc_sec(vis, 100.0);
    oskar_vis_set_channel_bandwidth_hz(vis, 0.15e6);
    oskar_vis_set_phase_centre(vis,
            ra0_rad * (180.0/M_PI), dec0_rad * (180.0/M_PI));

    oskar_vis_add_system_noise(vis, tel, seed, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);

    // Evaluate baseline coordinates
    settings.obs.num_pointing_levels = 1;
    settings.obs.phase_centre_lon_rad = (double*) malloc(sizeof(double));
    settings.obs.phase_centre_lat_rad = (double*) malloc(sizeof(double));
    settings.obs.phase_centre_lon_rad[0] = ra0_rad;
    settings.obs.phase_centre_lat_rad[0] = dec0_rad;
    settings.obs.start_frequency_hz = oskar_vis_freq_start_hz(vis);
    settings.obs.num_channels = num_channels;
    settings.obs.frequency_inc_hz = oskar_vis_freq_inc_hz(vis);
    settings.obs.num_time_steps = num_times;
    settings.obs.start_mjd_utc = oskar_vis_time_start_mjd_utc(vis);
    settings.obs.length_sec = num_times * oskar_vis_time_inc_sec(vis);
    settings.obs.length_days = settings.obs.length_sec / 86400.0;
    settings.obs.dt_dump_days = oskar_vis_time_inc_sec(vis) / 86400.0;

    oskar_Mem *work_uvw;
    work_uvw = oskar_mem_create(type, OSKAR_CPU, 3 * num_stations,
            &err);
    oskar_convert_ecef_to_baseline_uvw(num_stations,
            oskar_telescope_station_true_x_offset_ecef_metres_const(tel),
            oskar_telescope_station_true_y_offset_ecef_metres_const(tel),
            oskar_telescope_station_true_z_offset_ecef_metres_const(tel),
            ra0_rad, dec0_rad, settings.obs.num_time_steps,
            settings.obs.start_mjd_utc, settings.obs.dt_dump_days, 0,
            oskar_vis_baseline_uu_metres(vis),
            oskar_vis_baseline_vv_metres(vis),
            oskar_vis_baseline_ww_metres(vis), work_uvw, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);

    oskar_Image* image;
    settings.image.input_vis_data = NULL;
    settings.image.size = 256;
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


    image = oskar_make_image(0, vis, &(settings.image), &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);

    //    err = oskar_image_write(&image, NULL, settings.image.oskar_image, 0);
    //    ASSERT_EQ(0, err) << oskar_get_error_string(err);


#if 0
    check_image_stats(&image, telescope);
#endif

    oskar_mem_free(work_uvw, &err);
    oskar_telescope_free(tel, &err);
    oskar_vis_free(vis, &err);
    oskar_image_free(image, &err);
}


static void generate_range(oskar_Mem* data, int number,
        double start, double inc)
{
    int err = 0;
    oskar_mem_realloc(data, number, &err);
    ASSERT_EQ(0, err) << oskar_get_error_string(err);

    if (oskar_mem_type(data) == OSKAR_DOUBLE)
    {
        double* d_ = oskar_mem_double(data, &err);
        for (int i = 0; i < number; ++i)
            d_[i] = start + i * inc;
    }
    else
    {
        float* d_ = oskar_mem_float(data, &err);
        for (int i = 0; i < number; ++i)
            d_[i] = start + i * inc;
    }
}


#if 0
static void check_image_stats(oskar_Image* image, oskar_Telescope* tel)
{
    int num_pixels = image->width * image->height;
    int type = oskar_mem_type(&image->data);
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
    oskar_Mem* s = &oskar_telescope_station(tel, 0)->noise.rms;
    int num_baselines = oskar_telescope_num_baselines(tel);

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
#endif
