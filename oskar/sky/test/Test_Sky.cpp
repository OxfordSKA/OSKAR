/*
 * Copyright (c) 2011-2019, The University of Oxford
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

#include "telescope/oskar_telescope.h"
#include "sky/oskar_sky.h"
#include "convert/oskar_convert_lon_lat_to_relative_directions.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_timer.h"
#include "utility/oskar_device.h"

#include <cstdlib>
#include "math/oskar_cmath.h"

#ifdef OSKAR_HAVE_CUDA
static int device_loc = OSKAR_GPU;
#else
static int device_loc = OSKAR_CPU;
#endif

TEST(oskar_Sky, copy)
{
    int status = 0;

    // Create and fill sky model 1.
    int sky1_num_sources = 50e3;
    oskar_Sky* sky1 = oskar_sky_create(OSKAR_SINGLE,
            OSKAR_CPU, sky1_num_sources, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < sky1_num_sources; ++i)
    {
        double value = (double)i;
        oskar_sky_set_source(sky1, i, value, value,
                value, value, value, value,
                value, value, 0.0, 0.0, 0.0, 0.0, &status);
    }
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Create sky model 2
    int sky2_num_sorces = 50e3;
    oskar_Sky* sky2 = oskar_sky_create(OSKAR_SINGLE,
            device_loc, sky2_num_sorces, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy sky model 1 into 2
//    oskar_Timer *t = oskar_timer_create(OSKAR_TIMER_CUDA);
//    oskar_timer_start(t);
//    for (int i = 0; i < 200; ++i)
//    {
    oskar_sky_copy(sky2, sky1, &status);
//    }
//    printf("Time taken = %f ms\n", (oskar_timer_elapsed(t)/200.0)*1000.0);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy back and check contents.
    oskar_Sky* sky_temp = oskar_sky_create_copy(sky2, OSKAR_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(sky1_num_sources, oskar_sky_num_sources(sky_temp));
    for (int i = 0; i < oskar_sky_num_sources(sky_temp); ++i)
    {
        if (i < sky1_num_sources)
        {
            EXPECT_FLOAT_EQ((float)i,
                    oskar_mem_float(oskar_sky_ra_rad(sky_temp), &status)[i]);
        }
        else
        {
            EXPECT_FLOAT_EQ((float)(i - sky1_num_sources) + 0.5,
                    oskar_mem_float(oskar_sky_ra_rad(sky_temp), &status)[i]);
        }
    }

    // Free memory.
    oskar_sky_free(sky_temp, &status);
    oskar_sky_free(sky1, &status);
    oskar_sky_free(sky2, &status);

}


TEST(SkyModel, append)
{
    int status = 0;

    // Create and fill sky model 1.
    int sky1_num_sources = 2;
    oskar_Sky* sky1 = oskar_sky_create(OSKAR_SINGLE,
            device_loc, sky1_num_sources, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < sky1_num_sources; ++i)
    {
        double value = (double)i;
        oskar_sky_set_source(sky1, i, value, value,
                value, value, value, value,
                value, value, 0.0, 0.0, 0.0, 0.0, &status);
    }
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Create and fill sky model 2.
    int sky2_num_sorces = 3;
    oskar_Sky* sky2 = oskar_sky_create(OSKAR_SINGLE,
            OSKAR_CPU, sky2_num_sorces, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < sky2_num_sorces; ++i)
    {
        double value = (double)i + 0.5;
        oskar_sky_set_source(sky2, i, value, value,
                value, value, value, value,
                value, value, 0.0, 0.0, 0.0, 0.0, &status);
    }
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Append sky2 to sky1.
    oskar_sky_append(sky1, sky2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(device_loc, oskar_sky_mem_location(sky1));

    // Copy back and check contents.
    oskar_Sky* sky_temp = oskar_sky_create_copy(sky1, OSKAR_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(sky1_num_sources + sky2_num_sorces,
            oskar_sky_num_sources(sky_temp));
    for (int i = 0; i < oskar_sky_num_sources(sky_temp); ++i)
    {
        if (i < sky1_num_sources)
        {
            EXPECT_FLOAT_EQ((float)i,
                    oskar_mem_float(oskar_sky_ra_rad(sky_temp), &status)[i]);
        }
        else
        {
            EXPECT_FLOAT_EQ((float)(i - sky1_num_sources) + 0.5,
                    oskar_mem_float(oskar_sky_ra_rad(sky_temp), &status)[i]);
        }
    }

    // Free memory.
    oskar_sky_free(sky_temp, &status);
    oskar_sky_free(sky1, &status);
    oskar_sky_free(sky2, &status);
}


TEST(SkyModel, compute_relative_lmn)
{
    int status = 0;
    const float deg2rad = 0.0174532925199432957692f;

    // Create some sources.
    float ra[] = {30.0f, 45.0f};
    float dec[] = {50.0f, 60.0f};
    int n = sizeof(ra) / sizeof(float);
    for (int i = 0; i < n; ++i)
    {
        ra[i] *= deg2rad;
        dec[i] *= deg2rad;
    }

    // Define phase centre.
    float ra0 = 30.0f * deg2rad;
    float dec0 = 55.0f * deg2rad;

    // Construct a sky model on the GPU.
    oskar_Sky* sky1 = oskar_sky_create(OSKAR_SINGLE, device_loc, n, &status);
    ASSERT_EQ(n, oskar_sky_num_sources(sky1));

    // Set values of these sources.
    for (int i = 0; i < n; ++i)
    {
        oskar_sky_set_source(sky1, i, ra[i], dec[i], 1.0, 2.0, 3.0, 4.0,
                200.0e6, -0.7, 0.0, 0.0, 0.0, 0.0, &status);
    }
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Compute direction cosines.
    oskar_sky_evaluate_relative_directions(sky1, ra0, dec0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy data back to CPU.
    oskar_Sky* sky2 = oskar_sky_create_copy(sky1, OSKAR_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_sky_free(sky1, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check the data.
    float tol = (float) 1e-6;
    for (int i = 0; i < n; ++i)
    {
        float l = sin(ra[i] - ra0) * cos(dec[i]);
        float m = cos(dec0) * sin(dec[i]) -
                sin(dec0) * cos(dec[i]) * cos(ra[i] - ra0);
        float p = sqrt(1.0 - l*l - m*m);
        EXPECT_NEAR(l, oskar_mem_float(oskar_sky_l(sky2), &status)[i], tol);
        EXPECT_NEAR(m, oskar_mem_float(oskar_sky_m(sky2), &status)[i], tol);
        EXPECT_NEAR(p, oskar_mem_float(oskar_sky_n(sky2), &status)[i], tol);
    }
    oskar_sky_free(sky2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}


TEST(SkyModel, copy_contents)
{
    int src_size = 20, dst_size = 60, status = 0;

    oskar_Sky *dst = 0, *src = 0;
    dst = oskar_sky_create(OSKAR_DOUBLE, OSKAR_CPU, dst_size, &status);
    src = oskar_sky_create(OSKAR_DOUBLE, OSKAR_CPU, src_size, &status);
    for (int i = 0; i < src_size; ++i)
    {
        oskar_sky_set_source(src, i,
                i + 0.0, i + 0.1, i + 0.2, i + 0.3, i + 0.4, i + 0.5,
                i + 0.6, i + 0.7, i + 0.8, i + 0.9, i + 1.0, i + 1.1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    oskar_sky_copy_contents(dst, src, 0 * src_size, 0, src_size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_sky_copy_contents(dst, src, 1 * src_size, 0, src_size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_sky_copy_contents(dst, src, 2 * src_size, 0, src_size, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    double* ra   = oskar_mem_double(oskar_sky_ra_rad(dst), &status);
    double* dec  = oskar_mem_double(oskar_sky_dec_rad(dst), &status);
    double* I    = oskar_mem_double(oskar_sky_I(dst), &status);
    double* Q    = oskar_mem_double(oskar_sky_Q(dst), &status);
    double* U    = oskar_mem_double(oskar_sky_U(dst), &status);
    double* V    = oskar_mem_double(oskar_sky_V(dst), &status);
    double* ref  = oskar_mem_double(oskar_sky_reference_freq_hz(dst), &status);
    double* spix = oskar_mem_double(oskar_sky_spectral_index(dst), &status);
    double* rm   = oskar_mem_double(oskar_sky_rotation_measure_rad(dst), &status);
    double* maj  = oskar_mem_double(oskar_sky_fwhm_major_rad(dst), &status);
    double* min  = oskar_mem_double(oskar_sky_fwhm_minor_rad(dst), &status);
    double* pa   = oskar_mem_double(oskar_sky_position_angle_rad(dst), &status);

    for (int j = 0, s = 0; j < 3; ++j)
    {
        for (int i = 0; i < src_size; ++i, ++s)
        {
            EXPECT_DOUBLE_EQ(i + 0.0, ra[s]);
            EXPECT_DOUBLE_EQ(i + 0.1, dec[s]);
            EXPECT_DOUBLE_EQ(i + 0.2, I[s]);
            EXPECT_DOUBLE_EQ(i + 0.3, Q[s]);
            EXPECT_DOUBLE_EQ(i + 0.4, U[s]);
            EXPECT_DOUBLE_EQ(i + 0.5, V[s]);
            EXPECT_DOUBLE_EQ(i + 0.6, ref[s]);
            EXPECT_DOUBLE_EQ(i + 0.7, spix[s]);
            EXPECT_DOUBLE_EQ(i + 0.8, rm[s]);
            EXPECT_DOUBLE_EQ(i + 0.9, maj[s]);
            EXPECT_DOUBLE_EQ(i + 1.0, min[s]);
            EXPECT_DOUBLE_EQ(i + 1.1, pa[s]);
        }
    }
    oskar_sky_free(src, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_sky_free(dst, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}


TEST(SkyModel, evaluate_gaussian_source_parameters)
{
    const double asec2rad = M_PI / (180.0 * 3600.0);
    const double deg2rad  = M_PI / 180.0;

    int num_sources = 16384;
    int status = 0;
    int num_failed = 0;

    oskar_Sky* sky = oskar_sky_create(OSKAR_DOUBLE,
            OSKAR_CPU, num_sources, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < num_sources; ++i)
    {
        oskar_sky_set_source(sky, i,
                i * deg2rad * 0.001, i * deg2rad * 0.001,
                0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                1200 * asec2rad, 600 * asec2rad, 30 * deg2rad, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }
    oskar_Timer* tmr = oskar_timer_create(OSKAR_TIMER_NATIVE);
    oskar_timer_start(tmr);
    oskar_sky_evaluate_gaussian_source_parameters(sky, 0,
            0.0, 10 * deg2rad, &num_failed, &status);
    printf("Evaluate Gaussian source parameters took %.3f s\n",
            oskar_timer_elapsed(tmr));
    oskar_timer_free(tmr);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_sky_free(sky, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(0, num_failed);
}


TEST(SkyModel, filter_by_radius)
{
    // Generate 91 sources from dec = 0 to dec = 90 degrees.
    int status = 0;
    int num_sources = 91;
    oskar_Sky* sky = oskar_sky_create(OSKAR_DOUBLE,
            OSKAR_CPU, num_sources, &status);
    for (int i = 0; i < num_sources; ++i)
    {
        oskar_sky_set_source(sky, i,
                0.0, i * ((M_PI / 2) / (num_sources - 1)),
                1.0 * i, 1.0, 2.0, 3.0, i * 100.0, i * 200.0, i * 300.0,
                0.0, 0.0, 0.0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    // Check that the data was set correctly.
    ASSERT_EQ(num_sources, oskar_sky_num_sources(sky));
    for (int i = 0; i < num_sources; ++i)
    {
        EXPECT_DOUBLE_EQ(0.0, oskar_mem_double(oskar_sky_ra_rad(sky), &status)[i]);
        EXPECT_DOUBLE_EQ(i * ((M_PI / 2) / (num_sources - 1)),
                oskar_mem_double(oskar_sky_dec_rad(sky), &status)[i]);
        EXPECT_DOUBLE_EQ(1.0 * i,
                oskar_mem_double(oskar_sky_I(sky), &status)[i]);
        EXPECT_DOUBLE_EQ(1.0, oskar_mem_double(oskar_sky_Q(sky), &status)[i]);
        EXPECT_DOUBLE_EQ(2.0, oskar_mem_double(oskar_sky_U(sky), &status)[i]);
        EXPECT_DOUBLE_EQ(3.0, oskar_mem_double(oskar_sky_V(sky), &status)[i]);
        EXPECT_DOUBLE_EQ(i * 100,
                oskar_mem_double(oskar_sky_reference_freq_hz(sky), &status)[i]);
        EXPECT_DOUBLE_EQ(i * 200,
                oskar_mem_double(oskar_sky_spectral_index(sky), &status)[i]);
    }

    // Filter the data.
    double inner = 4.5 * M_PI / 180;
    double outer = 10.5 * M_PI / 180;
    double ra0 = 0.0;
    double dec0 = M_PI / 2;
    oskar_sky_filter_by_radius(sky, inner, outer, ra0, dec0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check the resulting sky model.
    ASSERT_EQ(6, oskar_sky_num_sources(sky));
    double *dec_ = oskar_mem_double(oskar_sky_dec_rad(sky), &status);
    for (int i = 0; i < oskar_sky_num_sources(sky); ++i)
    {
        ASSERT_GT(dec_[i], 79.5 * M_PI / 180.0);
        ASSERT_LT(dec_[i], 85.5 * M_PI / 180.0);
    }

    // Free memory.
    oskar_sky_free(sky, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}


TEST(SkyModel, filter_by_flux)
{
    int i = 0, type = 0, num_sources = 223, status = 0;
    double flux_min = 5.0;
    double flux_max = 10.0;

    // Single precision.
    type = OSKAR_SINGLE;
    {
        // Create a test sky model.
        oskar_Sky* sky_input = oskar_sky_create(type,
                OSKAR_CPU, num_sources, &status);
        for (i = 0; i < num_sources; ++i)
        {
            oskar_sky_set_source(sky_input, i,
                    0.0, i * ((M_PI / 2) / (num_sources - 1)),
                    0.05 * i, 0.10 * i, 0.15 * i, 0.20 * i,
                    100.0 * i, 200.0 * i, 300.0 * i,
                    1000.0 * i, 2000.0 * i, 3000.0 * i,
                    &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }

        // Filter on CPU.
        oskar_Sky* sky_cpu = oskar_sky_create_copy(sky_input,
                OSKAR_CPU, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_sky_filter_by_flux(sky_cpu, flux_min, flux_max, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check that there are no sources with fluxes outside the range.
        float* I_cpu = oskar_mem_float(oskar_sky_I(sky_cpu), &status);
        for (i = 0; i < oskar_sky_num_sources(sky_cpu); ++i)
        {
            EXPECT_LE(I_cpu[i], flux_max) << "CPU flux filter failed: i=" << i;
            EXPECT_GE(I_cpu[i], flux_min) << "CPU flux filter failed: i=" << i;
        }

        // Free sky models.
        oskar_sky_free(sky_cpu, &status);
        oskar_sky_free(sky_input, &status);
    }

    // Double precision.
    type = OSKAR_DOUBLE;
    {
        // Create a test sky model.
        oskar_Sky* sky_input = oskar_sky_create(type,
                OSKAR_CPU, num_sources, &status);
        for (i = 0; i < num_sources; ++i)
        {
            oskar_sky_set_source(sky_input, i,
                    0.0, i * ((M_PI / 2) / (num_sources - 1)),
                    0.05 * i, 0.10 * i, 0.15 * i, 0.20 * i,
                    100.0 * i, 200.0 * i, 300.0 * i,
                    1000.0 * i, 2000.0 * i, 3000.0 * i,
                    &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }

        // Filter on CPU.
        oskar_Sky* sky_cpu = oskar_sky_create_copy(sky_input,
                OSKAR_CPU, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_sky_filter_by_flux(sky_cpu, flux_min, flux_max, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Check that there are no sources with fluxes outside the range.
        double* I_cpu = oskar_mem_double(oskar_sky_I(sky_cpu), &status);
        for (i = 0; i < oskar_sky_num_sources(sky_cpu); ++i)
        {
            EXPECT_LE(I_cpu[i], flux_max) << "CPU flux filter failed: i=" << i;
            EXPECT_GE(I_cpu[i], flux_min) << "CPU flux filter failed: i=" << i;
        }

        // Free sky models.
        oskar_sky_free(sky_cpu, &status);
        oskar_sky_free(sky_input, &status);
    }
}


void horizon_clip(const oskar_Sky* sky_in, const oskar_Telescope* telescope,
        int type, int location, int* status)
{
    // Create a work buffer and output sky model.
    oskar_StationWork* work = oskar_station_work_create(type, location, status);
    oskar_Sky* sky_out = oskar_sky_create(type, location, 0, status);
    oskar_Timer* tmr = oskar_timer_create(OSKAR_TIMER_NATIVE);

    // Copy input sky model to device location.
    const int n_sources = oskar_sky_num_sources(sky_in);
    oskar_Sky* sky_in_dev = oskar_sky_create_copy(sky_in, location, status);

    // Horizon clip should succeed.
    for (int i = 0; i < 1; ++i)
    {
        oskar_timer_start(tmr);
        oskar_sky_horizon_clip(sky_out, sky_in_dev, telescope, 0.0, work, status);
        printf("Horizon clip took %.3f s\n", oskar_timer_elapsed(tmr));
        ASSERT_EQ(0, *status) << oskar_get_error_string(*status);
        EXPECT_EQ(n_sources / 2, oskar_sky_num_sources(sky_out));
    }
    printf("Done.\n");

    // Check sky data.
    oskar_Sky* sky_temp = oskar_sky_create_copy(sky_out, OSKAR_CPU, status);
    EXPECT_EQ(n_sources / 2, oskar_sky_num_sources(sky_temp));
    const float* dec = oskar_mem_float_const(oskar_sky_dec_rad_const(sky_temp),
            status);
    ASSERT_EQ(0, *status) << oskar_get_error_string(*status);
    for (int i = 0, n = oskar_sky_num_sources(sky_temp); i < n; ++i)
    {
        EXPECT_GT(dec[i], 0.0f);
    }

    oskar_sky_free(sky_temp, status);
    oskar_sky_free(sky_in_dev, status);
    oskar_sky_free(sky_out, status);
    oskar_station_work_free(work, status);
    oskar_timer_free(tmr);
}


TEST(SkyModel, horizon_clip)
{
    int status = 0;
    int type = OSKAR_SINGLE;

    // Constants.
    const double deg2rad = M_PI / 180.0;

    // Sky grid parameters.
    int n_lat = 128; // 8
    int n_lon = 128; // 12
    int n_sources = n_lat * n_lon;
    double lat_start = -90.0;
    double lon_start = 0.0;
    double lat_end = 90.0;
    double lon_end = 330.0;

    // Generate grid.
    oskar_Sky* sky_in = oskar_sky_create(type, OSKAR_CPU, n_sources, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0, k = 0; i < n_lat; ++i)
    {
        for (int j = 0; j < n_lon; ++j, ++k)
        {
            double ra = lon_start + j * (lon_end - lon_start) / (n_lon - 1);
            double dec = lat_start + i * (lat_end - lat_start) / (n_lat - 1);
            oskar_sky_set_source(sky_in, k, ra * deg2rad, dec * deg2rad,
                    double(k), double(2*k), double(3*k), double(4*k),
                    double(5*k), double(6*k), double(7*k), 0.0, 0.0, 0.0,
                    &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
    }

    // Evaluate relative direction cosines.
    oskar_sky_evaluate_relative_directions(sky_in, 0.0, M_PI/2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Create a telescope model near the north pole.
    int n_stations = 512;
    oskar_Telescope* telescope = oskar_telescope_create(type,
            OSKAR_CPU, n_stations, &status);
    oskar_telescope_resize_station_array(telescope, n_stations, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < n_stations; ++i)
    {
        oskar_station_set_position(oskar_telescope_station(telescope, i),
                0.0, (90.0 - i * 0.0001) * deg2rad, 0.0, 0.0, 0.0, 0.0);
    }

    // Horizon clip on CPU.
    horizon_clip(sky_in, telescope, type, OSKAR_CPU, &status);

#ifdef OSKAR_HAVE_CUDA
    // Horizon clip on GPU.
    horizon_clip(sky_in, telescope, type, OSKAR_GPU, &status);
#endif

#ifdef OSKAR_HAVE_OPENCL
    // Horizon clip on OpenCL.
    char* device_name = oskar_device_name(OSKAR_CL, 0);
    printf("Using %s\n", device_name);
    free(device_name);
    horizon_clip(sky_in, telescope, type, OSKAR_CL, &status);
#endif

    oskar_sky_free(sky_in, &status);
    oskar_telescope_free(telescope, &status);
}


TEST(SkyModel, resize)
{
    int status = 0;

    // Resizing on the GPU in single precision
    {
        oskar_Sky* sky = oskar_sky_create(OSKAR_SINGLE,
                device_loc, 10, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ((int)OSKAR_SINGLE, oskar_sky_precision(sky));
        ASSERT_EQ(device_loc, oskar_sky_mem_location(sky));
        ASSERT_EQ(10, oskar_sky_num_sources(sky));
        oskar_sky_resize(sky, 1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(1, oskar_sky_num_sources(sky));
        oskar_sky_resize(sky, 20, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(20, oskar_sky_num_sources(sky));
        oskar_sky_free(sky, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    // Resizing on the CPU in double precision
    {
        oskar_Sky* sky = oskar_sky_create(OSKAR_DOUBLE,
                OSKAR_CPU, 10, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ((int)OSKAR_DOUBLE, oskar_sky_precision(sky));
        ASSERT_EQ((int)OSKAR_CPU, oskar_sky_mem_location(sky));
        ASSERT_EQ(10, oskar_sky_num_sources(sky));
        oskar_sky_resize(sky, 1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(1, oskar_sky_num_sources(sky));
        oskar_sky_resize(sky, 20, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ(20, oskar_sky_num_sources(sky));
        oskar_sky_free(sky, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }
}


TEST(SkyModel, scale_by_spectral_index)
{
    int num_sources = 10000, status = 0;
    oskar_Timer* timer = 0;
//    double sec = 0.0;
    double stokes_I = 10.0, stokes_Q = 1.0, stokes_U = 0.5, stokes_V = 0.1;
    double spix = -0.7, freq_ref = 10.0e6, freq_new = 50.0e6;

    // Create and fill a sky model.
    oskar_Sky* sky = oskar_sky_create(OSKAR_SINGLE, OSKAR_CPU,
            num_sources, &status);
    oskar_mem_set_value_real(oskar_sky_I(sky), stokes_I,
            0, num_sources, &status);
    oskar_mem_set_value_real(oskar_sky_Q(sky), stokes_Q,
            0, num_sources, &status);
    oskar_mem_set_value_real(oskar_sky_U(sky), stokes_U,
            0, num_sources, &status);
    oskar_mem_set_value_real(oskar_sky_V(sky), stokes_V,
            0, num_sources, &status);
    oskar_mem_set_value_real(oskar_sky_reference_freq_hz(sky), freq_ref,
            0, num_sources, &status);
    oskar_mem_set_value_real(oskar_sky_spectral_index(sky), spix,
            0, num_sources, &status);

    // Copy to GPU.
    oskar_Sky* sky_gpu = oskar_sky_create_copy(sky, device_loc, &status);

    // Scale on CPU.
    timer = oskar_timer_create(OSKAR_TIMER_NATIVE);
    oskar_timer_resume(timer);
    oskar_sky_scale_flux_with_frequency(sky, freq_new, &status);
//    sec = oskar_timer_elapsed(timer);
//    printf("Scale by spectral index (CPU): %.6f\n", sec);
    oskar_timer_free(timer);

    // Scale on GPU.
    timer = oskar_timer_create(OSKAR_TIMER_CUDA);
    oskar_timer_resume(timer);
    oskar_sky_scale_flux_with_frequency(sky_gpu, freq_new, &status);
//    sec = oskar_timer_elapsed(timer);
//    printf("Scale by spectral index (GPU): %.6f\n", sec);
    oskar_timer_free(timer);

    // Copy GPU data to CPU for check.
    oskar_Sky* sky_cpu = oskar_sky_create_copy(sky_gpu, OSKAR_CPU, &status);

    // Check contents.
    float* I_cpu = oskar_mem_float(oskar_sky_I(sky), &status);
    float* Q_cpu = oskar_mem_float(oskar_sky_Q(sky), &status);
    float* U_cpu = oskar_mem_float(oskar_sky_U(sky), &status);
    float* V_cpu = oskar_mem_float(oskar_sky_V(sky), &status);
    float* ref_cpu = oskar_mem_float(oskar_sky_reference_freq_hz(sky), &status);
    float* spx_cpu = oskar_mem_float(oskar_sky_spectral_index(sky), &status);
    float* I_gpu = oskar_mem_float(oskar_sky_I(sky_cpu), &status);
    float* Q_gpu = oskar_mem_float(oskar_sky_Q(sky_cpu), &status);
    float* U_gpu = oskar_mem_float(oskar_sky_U(sky_cpu), &status);
    float* V_gpu = oskar_mem_float(oskar_sky_V(sky_cpu), &status);
    float* ref_gpu = oskar_mem_float(oskar_sky_reference_freq_hz(sky_cpu), &status);
    float* spx_gpu = oskar_mem_float(oskar_sky_spectral_index(sky_cpu), &status);

    for (int i = 0; i < num_sources; ++i)
    {
        double factor = pow(freq_new / freq_ref, spix);
        ASSERT_FLOAT_EQ(stokes_I * factor, I_cpu[i]);
        ASSERT_FLOAT_EQ(stokes_I * factor, I_gpu[i]);
        ASSERT_FLOAT_EQ(stokes_Q * factor, Q_cpu[i]);
        ASSERT_FLOAT_EQ(stokes_Q * factor, Q_gpu[i]);
        ASSERT_FLOAT_EQ(stokes_U * factor, U_cpu[i]);
        ASSERT_FLOAT_EQ(stokes_U * factor, U_gpu[i]);
        ASSERT_FLOAT_EQ(stokes_V * factor, V_cpu[i]);
        ASSERT_FLOAT_EQ(stokes_V * factor, V_gpu[i]);
        ASSERT_FLOAT_EQ(freq_new, ref_cpu[i]);
        ASSERT_FLOAT_EQ(freq_new, ref_gpu[i]);
        ASSERT_FLOAT_EQ(spix, spx_cpu[i]);
        ASSERT_FLOAT_EQ(spix, spx_gpu[i]);
    }

    oskar_sky_free(sky, &status);
    oskar_sky_free(sky_gpu, &status);
    oskar_sky_free(sky_cpu, &status);
}

TEST(SkyModel, rotation_measure)
{
    int num_sources = 10000, status = 0;
    oskar_Timer* timer = 0;
//    double sec = 0.0;
    double stokes_I = 10.0, stokes_Q = 1.0, stokes_U = 0.0, stokes_V = 0.1;
    double spix = 0.0, freq_ref = 100.0e6, freq_new = 99e6, rm = 0.5;
    double max_err = 0.0, avg_err = 0.0;

    // Create and fill a sky model.
    oskar_Sky* sky_ref = oskar_sky_create(OSKAR_SINGLE, OSKAR_CPU,
            num_sources, &status);
    oskar_mem_set_value_real(oskar_sky_I(sky_ref), stokes_I,
            0, num_sources, &status);
    oskar_mem_set_value_real(oskar_sky_Q(sky_ref), stokes_Q,
            0, num_sources, &status);
    oskar_mem_set_value_real(oskar_sky_U(sky_ref), stokes_U,
            0, num_sources, &status);
    oskar_mem_set_value_real(oskar_sky_V(sky_ref), stokes_V,
            0, num_sources, &status);
    oskar_mem_set_value_real(oskar_sky_reference_freq_hz(sky_ref), freq_ref,
            0, num_sources, &status);
    oskar_mem_set_value_real(oskar_sky_spectral_index(sky_ref), spix,
            0, num_sources, &status);
    oskar_mem_set_value_real(oskar_sky_rotation_measure_rad(sky_ref), rm,
            0, num_sources, &status);

    // Copy to CPU.
    oskar_Sky* sky_cpu = oskar_sky_create_copy(sky_ref, OSKAR_CPU, &status);

    // Copy to GPU.
    oskar_Sky* sky_gpu = oskar_sky_create_copy(sky_ref, device_loc, &status);

    // Scale on CPU.
    timer = oskar_timer_create(OSKAR_TIMER_NATIVE);
    oskar_timer_resume(timer);
    oskar_sky_scale_flux_with_frequency(sky_cpu, freq_new, &status);
//    sec = oskar_timer_elapsed(timer);
//    printf("Rotation measure scaling (CPU): %.6f\n", sec);
    oskar_timer_free(timer);

    // Scale on GPU.
    timer = oskar_timer_create(OSKAR_TIMER_CUDA);
    oskar_timer_resume(timer);
    oskar_sky_scale_flux_with_frequency(sky_gpu, freq_new, &status);
//    sec = oskar_timer_elapsed(timer);
//    printf("Rotation measure scaling (GPU): %.6f\n", sec);
    oskar_timer_free(timer);

//    oskar_sky_save("sky1.osm", sky_cpu, &status);
    EXPECT_EQ(0, status) << oskar_get_error_string(status);

    // Check contents for consistency.
    oskar_mem_evaluate_relative_error(oskar_sky_I(sky_gpu),
            oskar_sky_I(sky_cpu), 0, &max_err, &avg_err, 0, &status);
    EXPECT_EQ(0, status);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(avg_err, 1e-6);
    oskar_mem_evaluate_relative_error(oskar_sky_Q(sky_gpu),
            oskar_sky_Q(sky_cpu), 0, &max_err, &avg_err, 0, &status);
    EXPECT_EQ(0, status);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(avg_err, 1e-6);
    oskar_mem_evaluate_relative_error(oskar_sky_U(sky_gpu),
            oskar_sky_U(sky_cpu), 0, &max_err, &avg_err, 0, &status);
    EXPECT_EQ(0, status);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(avg_err, 1e-6);
    oskar_mem_evaluate_relative_error(oskar_sky_V(sky_gpu),
            oskar_sky_V(sky_cpu), 0, &max_err, &avg_err, 0, &status);
    EXPECT_EQ(0, status);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(avg_err, 1e-6);
    oskar_mem_evaluate_relative_error(oskar_sky_reference_freq_hz(sky_gpu),
            oskar_sky_reference_freq_hz(sky_cpu), 0, &max_err, &avg_err, 0, &status);
    EXPECT_EQ(0, status);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(avg_err, 1e-6);

    // Scale back to reference frequency on CPU.
    oskar_sky_scale_flux_with_frequency(sky_cpu, freq_ref, &status);

    // Scale back to reference frequency on GPU.
    oskar_sky_scale_flux_with_frequency(sky_gpu, freq_ref, &status);

//    oskar_sky_save("sky2.osm", sky_cpu, &status);
    EXPECT_EQ(0, status) << oskar_get_error_string(status);

    // Check contents for consistency.
    oskar_mem_evaluate_relative_error(oskar_sky_I(sky_gpu),
            oskar_sky_I(sky_cpu), 0, &max_err, &avg_err, 0, &status);
    EXPECT_EQ(0, status);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(avg_err, 1e-6);
    oskar_mem_evaluate_relative_error(oskar_sky_Q(sky_gpu),
            oskar_sky_Q(sky_cpu), 0, &max_err, &avg_err, 0, &status);
    EXPECT_EQ(0, status);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(avg_err, 1e-6);
    oskar_mem_evaluate_relative_error(oskar_sky_U(sky_gpu),
            oskar_sky_U(sky_cpu), 0, &max_err, &avg_err, 0, &status);
    EXPECT_EQ(0, status);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(avg_err, 1e-6);
    oskar_mem_evaluate_relative_error(oskar_sky_V(sky_gpu),
            oskar_sky_V(sky_cpu), 0, &max_err, &avg_err, 0, &status);
    EXPECT_EQ(0, status);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(avg_err, 1e-6);
    oskar_mem_evaluate_relative_error(oskar_sky_reference_freq_hz(sky_gpu),
            oskar_sky_reference_freq_hz(sky_cpu), 0, &max_err, &avg_err, 0, &status);
    EXPECT_EQ(0, status);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(avg_err, 1e-6);

    // Check contents for consistency with original reference sky.
    oskar_mem_evaluate_relative_error(oskar_sky_I(sky_cpu),
            oskar_sky_I(sky_ref), 0, &max_err, &avg_err, 0, &status);
    EXPECT_EQ(0, status);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(avg_err, 1e-6);
    oskar_mem_evaluate_relative_error(oskar_sky_Q(sky_cpu),
            oskar_sky_Q(sky_ref), 0, &max_err, &avg_err, 0, &status);
    EXPECT_EQ(0, status);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(avg_err, 1e-6);
    oskar_mem_evaluate_relative_error(oskar_sky_U(sky_cpu),
            oskar_sky_U(sky_ref), 0, &max_err, &avg_err, 0, &status);
    EXPECT_EQ(0, status);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(avg_err, 1e-6);
    oskar_mem_evaluate_relative_error(oskar_sky_V(sky_cpu),
            oskar_sky_V(sky_ref), 0, &max_err, &avg_err, 0, &status);
    EXPECT_EQ(0, status);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(avg_err, 1e-6);
    oskar_mem_evaluate_relative_error(oskar_sky_reference_freq_hz(sky_cpu),
            oskar_sky_reference_freq_hz(sky_ref), 0, &max_err, &avg_err, 0, &status);
    EXPECT_EQ(0, status);
    EXPECT_LT(max_err, 1e-6);
    EXPECT_LT(avg_err, 1e-6);

    oskar_sky_free(sky_ref, &status);
    oskar_sky_free(sky_gpu, &status);
    oskar_sky_free(sky_cpu, &status);
}


TEST(SkyModel, set_source)
{
    int status = 0;

    // Construct a sky model of zero size.
    oskar_Sky* sky = oskar_sky_create(OSKAR_SINGLE, OSKAR_CPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(0, oskar_sky_num_sources(sky));

    // Try to set a source into the model - this should fail as the model is
    // still zero size.
    oskar_sky_set_source(sky, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            200.0e6, -0.7, 2.0, 0.0, 0.0, 0.0, &status);
    ASSERT_EQ((int)OSKAR_ERR_OUT_OF_RANGE, status);
    status = 0;

    // Resize the model to 2 sources.
    oskar_sky_resize(sky, 2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(2, oskar_sky_num_sources(sky));

    // Set values of these 2 sources.
    oskar_sky_set_source(sky, 0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0,
            200.0e6, -0.7, 2.0, 0.0, 0.0, 0.0, &status);
    oskar_sky_set_source(sky, 1, 1.5, 2.5, 3.5, 4.5, 5.5, 6.5,
            250.0e6, -0.8, 2.5, 0.0, 0.0, 0.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ((int)OSKAR_SINGLE, oskar_sky_precision(sky));

    // Copy back into temp. structure and check the values were set correctly.
    oskar_Sky* sky_temp = oskar_sky_create_copy(sky, OSKAR_CPU, &status);
    ASSERT_EQ((int)OSKAR_CPU, oskar_sky_mem_location(sky_temp));
    ASSERT_EQ((int)OSKAR_SINGLE, oskar_sky_precision(sky_temp));
    ASSERT_EQ(2, oskar_sky_num_sources(sky_temp));
    EXPECT_FLOAT_EQ(1.0f, oskar_mem_float(oskar_sky_ra_rad(sky_temp), &status)[0]);
    EXPECT_FLOAT_EQ((float)200e6,
            oskar_mem_float(oskar_sky_reference_freq_hz(sky_temp), &status)[0]);
    EXPECT_FLOAT_EQ(4.5f, oskar_mem_float(oskar_sky_Q(sky_temp), &status)[1]);
    EXPECT_FLOAT_EQ(-0.8f, oskar_mem_float(oskar_sky_spectral_index(sky_temp),
            &status)[1]);
    EXPECT_FLOAT_EQ(2.5f, oskar_mem_float(oskar_sky_rotation_measure_rad(sky_temp),
            &status)[1]);

    // Free memory.
    oskar_sky_free(sky, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_sky_free(sky_temp, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}


TEST(SkyModel, sky_model_set)
{
    int status = 0;
    int number = 0;
    oskar_Sky** set = 0;
    int max_sources_per_model = 5;

    int type = OSKAR_DOUBLE;
    int location = OSKAR_CPU;

    int size1 = 6;
    oskar_Sky* model1 = oskar_sky_create(type, location,
            size1, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < size1; ++i)
    {
        oskar_mem_double(oskar_sky_ra_rad(model1), &status)[i] = 1.0 * i;
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }
    oskar_sky_append_to_set(&number, &set,
            max_sources_per_model, model1, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    int size2 = 7;
    oskar_Sky* model2 = oskar_sky_create(type, location,
            size2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    for (int i = 0; i < size2; ++i)
    {
        oskar_mem_double(oskar_sky_ra_rad(model2), &status)[i] = 1.0 * i + 0.5;
        oskar_mem_double(oskar_sky_fwhm_major_rad(model2), &status)[i] = i * 0.75;
    }
    oskar_sky_append_to_set(&number, &set,
            max_sources_per_model, model2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Free the individual sky models.
    oskar_sky_free(model1, &status);
    oskar_sky_free(model2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check the number of sets.
    EXPECT_EQ(3, number);

    // Check the contents of the set.
    for (int i = 0, s = 0; i < number; ++i)
    {
        int num_sources = oskar_sky_num_sources(set[i]);
        double* ra  = oskar_mem_double(oskar_sky_ra_rad(set[i]), &status);
        double* maj = oskar_mem_double(oskar_sky_fwhm_major_rad(set[i]), &status);
//        printf("++ set[%i] no. sources = %i, use extended = %s\n",
//                i, num_sources, set[i].use_extended ? "true" : "false");
        if (i != number - 1)
        {
            EXPECT_EQ(max_sources_per_model, num_sources);
        }
        for (int j = 0; j < num_sources; ++j, ++s)
        {
            if (s < size1)
            {
                EXPECT_DOUBLE_EQ(1.0 * s, ra[j]);
                EXPECT_DOUBLE_EQ(0.0, maj[j]);
            }
            else
            {
                EXPECT_DOUBLE_EQ(1.0 * (s-size1) + 0.5, ra[j]);
                EXPECT_DOUBLE_EQ((s-size1) * 0.75, maj[j]);
            }
        }
    }

    // Free the array of sky models.
    if (set)
    {
        for (int i = 0; i < number; ++i)
        {
            oskar_sky_free(set[i], &status);
            ASSERT_EQ(0, status) << oskar_get_error_string(status);
        }
        free(set);
    }
}


#if 0
TEST(SkyModel, test_gaussian_source)
{
    double ra0  = 0.0  * M_PI/180;
    double dec0 = 90.0 * M_PI/180;

    double ra       = 0.0  * (M_PI / 180.0);
    double dec      = 70.0 * (M_PI / 180.0);
    double fwhm_maj = 1.0  * (M_PI / 180.0);
    double fwhm_min = 1.0  * (M_PI / 180.0);
    double pa       = 0.0 * (M_PI / 180.0);

    double delta_ra_maj, delta_dec_maj, delta_ra_min, delta_dec_min;
    double lon[4], lat[4];

    delta_ra_maj  = (fwhm_maj / 2.0) * sin(pa);
    delta_dec_maj = (fwhm_maj / 2.0) * cos(pa);

    delta_ra_min  = (fwhm_min / 2.0) * cos(pa);
    delta_dec_min = (fwhm_min / 2.0) * sin(pa);

    lon[0] = ra - delta_ra_maj;
    lon[1] = ra + delta_ra_maj;
    lon[2] = ra - delta_ra_min;
    lon[3] = ra + delta_ra_min;

    lat[0] = dec - delta_dec_maj;
    lat[1] = dec + delta_dec_maj;
    lat[2] = dec - delta_dec_min;
    lat[3] = dec + delta_dec_min;

    double l[4], m[4], n[4];

    oskar_convert_lon_lat_to_relative_directions_d(4, lon, lat,
            ra0, dec0, l, m, n);

    printf("\n");
    printf("ra0, dec0              = %f, %f\n", ra0*(180.0/M_PI), dec0*(180.0/M_PI));
    printf("ra, dec                = %f, %f\n", ra*180/M_PI, dec*180/M_PI);
    printf("fwhm_maj, fwhm_min, pa = %f, %f, %f\n", fwhm_maj*180/M_PI,
            fwhm_min*180/M_PI, pa*180/M_PI);
    printf("delta ra (maj, min)    = %f, %f\n",
            delta_ra_maj*180/M_PI, delta_ra_min*180/M_PI);
    printf("delta dec (maj, min)   = %f, %f\n",
            delta_dec_maj*180/M_PI, delta_dec_min*180/M_PI);
    printf("\n");


    double x_maj = l[1] - l[0];
    double y_maj = m[1] - m[0];
    double pa_lm_maj = M_PI/2.0 - atan2(y_maj, x_maj);
    double fwhm_lm_maj = sqrt(pow(fabs(x_maj), 2.0) + pow(fabs(y_maj), 2.0));

    double x_min = l[3] - l[2];
    double y_min = m[3] - m[2];
    double pa_lm_min = M_PI/2.0 - atan2(y_min, x_min);
    double fwhm_lm_min = sqrt(pow(fabs(x_min), 2.0) + pow(fabs(y_min), 2.0));


    printf("= major axis:\n");
    printf("    lon, lat = %f->%f, %f->%f\n",
            lon[0]*(180/M_PI), lon[1]*(180/M_PI),
            lat[0]*(180/M_PI), lat[1]*(180/M_PI));
    printf("    l,m      = %f->%f, %f->%f\n", l[0], l[1], m[0], m[1]);
    printf("    x,y      = %f, %f\n", x_maj, y_maj);
    printf("    pa_lm    = %f\n", pa_lm_maj * (180.0/M_PI));
    printf("    fwhm     = %f\n", asin(fwhm_lm_maj)*180/M_PI);

    printf("= minor axis:\n");
    printf("    lon, lat = %f->%f, %f->%f\n",
            lon[2]*(180/M_PI), lon[3]*(180/M_PI),
            lat[2]*(180/M_PI), lat[3]*(180/M_PI));
    printf("    l,m      = %f->%f, %f->%f\n", l[2], l[3], m[2], m[3]);
    printf("    x,y      = %f, %f\n", x_min, y_min);
    printf("    pa_lm    = %f\n", pa_lm_min * (180.0/M_PI));
    printf("    fwhm     = %f\n", asin(fwhm_lm_min)*180/M_PI);
}
#endif


TEST(SkyModel, load_ascii)
{
    int status = 0;
    const float deg2rad = 0.0174532925199432957692f;
    const char* filename = "temp_sources.osm";

    // Load sky file with all columns specified.
    {
        FILE* file = fopen(filename, "w");
        if (!file) FAIL() << "Unable to create test file";
        int num_sources = 1013;
        for (int i = 0; i < num_sources; ++i)
        {
            if (i % 10 == 0) fprintf(file, "# some comment!\n");
            fprintf(file, "%f %f %f %f %f %f %f %f\n",
                    i/10.0, i/20.0, 0.0, 1.0, 2.0, 3.0, 200.0e6, -0.7);
        }
        fclose(file);

        // Load the file.
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_Sky* sky = oskar_sky_load(filename, OSKAR_SINGLE, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        ASSERT_EQ((int)OSKAR_SINGLE, oskar_sky_precision(sky));
        ASSERT_EQ((int)OSKAR_CPU, oskar_sky_mem_location(sky));
        ASSERT_EQ(num_sources, oskar_sky_num_sources(sky));

        // Check the data loaded correctly.
        for (int i = 0; i < num_sources; ++i)
        {
            ASSERT_FLOAT_EQ(i/10.0f * deg2rad,
                    oskar_mem_float(oskar_sky_ra_rad(sky), &status)[i]);
            ASSERT_FLOAT_EQ(i/20.0f * deg2rad,
                    oskar_mem_float(oskar_sky_dec_rad(sky), &status)[i]);
            ASSERT_FLOAT_EQ(0.0f, oskar_mem_float(oskar_sky_I(sky), &status)[i]);
            ASSERT_FLOAT_EQ(1.0f, oskar_mem_float(oskar_sky_Q(sky), &status)[i]);
            ASSERT_FLOAT_EQ(2.0f, oskar_mem_float(oskar_sky_U(sky), &status)[i]);
            ASSERT_FLOAT_EQ(3.0f, oskar_mem_float(oskar_sky_V(sky), &status)[i]);
            ASSERT_FLOAT_EQ((float)200.0e6,
                    oskar_mem_float(oskar_sky_reference_freq_hz(sky), &status)[i]);
            ASSERT_FLOAT_EQ(-0.7f,
                    oskar_mem_float(oskar_sky_spectral_index(sky), &status)[i]);
        }

        // Cleanup.
        oskar_sky_free(sky, &status);
        remove(filename);
    }

    // Load sky file with with just RA, Dec and I specified.
    {
        FILE* file = fopen(filename, "w");
        if (!file) FAIL() << "Unable to create test file";
        int num_sources = 1013;
        for (int i = 0; i < num_sources; ++i)
        {
            if (i % 10 == 0) fprintf(file, "# some comment!\n");
            fprintf(file, "%f, %f, %f\n", i/10.0, i/20.0, (float)i);
        }
        fclose(file);

        // Load the sky model.
        oskar_Sky* sky = oskar_sky_load(filename, OSKAR_SINGLE, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        ASSERT_EQ((int)OSKAR_SINGLE, oskar_sky_precision(sky));
        ASSERT_EQ((int)OSKAR_CPU, oskar_sky_mem_location(sky));
        ASSERT_EQ(num_sources, oskar_sky_num_sources(sky));

        // Check the data is correct.
        for (int i = 0; i < num_sources; ++i)
        {
            ASSERT_FLOAT_EQ(i/10.0f * deg2rad,
                    oskar_mem_float(oskar_sky_ra_rad(sky), &status)[i]);
            ASSERT_FLOAT_EQ(i/20.0f * deg2rad,
                    oskar_mem_float(oskar_sky_dec_rad(sky), &status)[i]);
            ASSERT_FLOAT_EQ((float)i,
                    oskar_mem_float(oskar_sky_I(sky), &status)[i]);
            ASSERT_FLOAT_EQ(0.0f, oskar_mem_float(oskar_sky_Q(sky), &status)[i]);
            ASSERT_FLOAT_EQ(0.0f, oskar_mem_float(oskar_sky_U(sky), &status)[i]);
            ASSERT_FLOAT_EQ(0.0f, oskar_mem_float(oskar_sky_V(sky), &status)[i]);
            ASSERT_FLOAT_EQ(0.0f,
                    oskar_mem_float(oskar_sky_reference_freq_hz(sky), &status)[i]);
            ASSERT_FLOAT_EQ(0.0f,
                    oskar_mem_float(oskar_sky_spectral_index(sky), &status)[i]);
        }

        // Cleanup.
        oskar_sky_free(sky, &status);
        remove(filename);
    }
}


TEST(SkyModel, read_write)
{
    oskar_Sky *sky = 0, *sky2 = 0;
    int type = OSKAR_DOUBLE;
    int location = OSKAR_CPU;
    int status = 0;
    int num_sources = 12345;
    sky = oskar_sky_create(type, location, num_sources, &status);
    const char* filename = "test_sky_model_write.osm";

    // Fill sky model with some test data.
    for (int i = 0; i < num_sources; ++i)
    {
        double ra = 1.0 * i;
        double dec = 2.1 * i;
        double I = 3.2 * i;
        double Q = 4.3 * i;
        double U = 5.4 * i;
        double V = 6.5 * i;
        double freq0 = 7.6 * i;
        double spix = 8.7 * i;
        double rm = 8.9 * i;
        double maj = 9.8 * i;
        double min = 10.9 * i;
        double pa = 11.1 * i;
        oskar_sky_set_source(sky, i, ra, dec, I, Q, U, V,
                freq0, spix, rm, maj, min, pa, &status);
    }
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Write it to a file.
    oskar_sky_write(sky, filename, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Read the data file into a new sky model structure.
    sky2 = oskar_sky_read(filename, location, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check the contents of the sky model.
    ASSERT_EQ(oskar_sky_num_sources(sky),
            oskar_sky_num_sources(sky2));
    double max_ = 0.0, avg_ = 0.0, tol = 1e-15;
    oskar_mem_evaluate_relative_error(oskar_sky_ra_rad_const(sky),
            oskar_sky_ra_rad_const(sky2), 0, &max_, &avg_, 0, &status);
    EXPECT_LT(max_, tol);
    EXPECT_LT(avg_, tol);
    oskar_mem_evaluate_relative_error(oskar_sky_dec_rad_const(sky),
            oskar_sky_dec_rad_const(sky2), 0, &max_, &avg_, 0, &status);
    EXPECT_LT(max_, tol);
    EXPECT_LT(avg_, tol);
    oskar_mem_evaluate_relative_error(oskar_sky_I_const(sky),
            oskar_sky_I_const(sky2), 0, &max_, &avg_, 0, &status);
    EXPECT_LT(max_, tol);
    EXPECT_LT(avg_, tol);
    oskar_mem_evaluate_relative_error(oskar_sky_Q_const(sky),
            oskar_sky_Q_const(sky2), 0, &max_, &avg_, 0, &status);
    EXPECT_LT(max_, tol);
    EXPECT_LT(avg_, tol);
    oskar_mem_evaluate_relative_error(oskar_sky_U_const(sky),
            oskar_sky_U_const(sky2), 0, &max_, &avg_, 0, &status);
    EXPECT_LT(max_, tol);
    EXPECT_LT(avg_, tol);
    oskar_mem_evaluate_relative_error(oskar_sky_V_const(sky),
            oskar_sky_V_const(sky2), 0, &max_, &avg_, 0, &status);
    EXPECT_LT(max_, tol);
    EXPECT_LT(avg_, tol);
    oskar_mem_evaluate_relative_error(oskar_sky_reference_freq_hz_const(sky),
            oskar_sky_reference_freq_hz_const(sky2), 0, &max_, &avg_, 0,
            &status);
    EXPECT_LT(max_, tol);
    EXPECT_LT(avg_, tol);
    oskar_mem_evaluate_relative_error(oskar_sky_spectral_index_const(sky),
            oskar_sky_spectral_index_const(sky2), 0, &max_, &avg_, 0,
            &status);
    EXPECT_LT(max_, tol);
    EXPECT_LT(avg_, tol);
    oskar_mem_evaluate_relative_error(oskar_sky_fwhm_major_rad_const(sky),
            oskar_sky_fwhm_major_rad_const(sky2), 0, &max_, &avg_, 0,
            &status);
    EXPECT_LT(max_, tol);
    EXPECT_LT(avg_, tol);
    oskar_mem_evaluate_relative_error(oskar_sky_fwhm_minor_rad_const(sky),
            oskar_sky_fwhm_minor_rad_const(sky2), 0, &max_, &avg_, 0,
            &status);
    EXPECT_LT(max_, tol);
    EXPECT_LT(avg_, tol);
    oskar_mem_evaluate_relative_error(oskar_sky_position_angle_rad_const(sky),
            oskar_sky_position_angle_rad_const(sky2), 0, &max_, &avg_, 0,
            &status);
    EXPECT_LT(max_, tol);
    EXPECT_LT(avg_, tol);

    // Free memory in both sky model structures.
    oskar_sky_free(sky2, &status);
    oskar_sky_free(sky, &status);

    // Remove the data file.
    remove(filename);
}
