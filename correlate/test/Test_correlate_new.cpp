/*
 * Copyright (c) 2015, The University of Oxford
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
#include <cuda_runtime_api.h>

#include <oskar_timer.h>

#include <oskar_correlate.h>
#include <oskar_correlate_point_time_smearing_cuda.h>
#include <oskar_correlate_point_time_smearing_new_cuda.h>
#include <oskar_get_error_string.h>
#include <cstdlib>

#define ALLOW_PRINTING 1

class correlate_new : public ::testing::Test
{
protected:
    static const int num_sources = 10000;
    static const int num_stations = 48;
    static const double bandwidth;
    oskar_Mem *u_, *v_, *w_;
    oskar_Telescope* tel;
    oskar_Sky* sky;
    oskar_Jones* jones;

protected:
    void createTestData(int precision, int location, int matrix)
    {
        int status = 0, type;

        // Allocate memory for data structures.
        type = precision | OSKAR_COMPLEX;
        if (matrix) type |= OSKAR_MATRIX;
        jones = oskar_jones_create(type, location, num_stations, num_sources,
                &status);
        u_ = oskar_mem_create(precision, location, num_stations, &status);
        v_ = oskar_mem_create(precision, location, num_stations, &status);
        w_ = oskar_mem_create(precision, location, num_stations, &status);
        sky = oskar_sky_create(precision, location, num_sources, &status);
        tel = oskar_telescope_create(precision, location,
                num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Fill data structures with random data in sensible ranges.
        srand(2);
        oskar_mem_random_range(oskar_jones_mem(jones), 1.0, 5.0, &status);
        oskar_mem_random_range(u_, 1.0, 5.0, &status);
        oskar_mem_random_range(v_, 1.0, 5.0, &status);
        oskar_mem_random_range(w_, 1.0, 5.0, &status);
        oskar_mem_random_range(
                oskar_telescope_station_true_x_offset_ecef_metres(tel),
                1.0, 10.0, &status);
        oskar_mem_random_range(
                oskar_telescope_station_true_y_offset_ecef_metres(tel),
                1.0, 10.0, &status);
        oskar_mem_random_range(
                oskar_telescope_station_true_z_offset_ecef_metres(tel),
                1.0, 10.0, &status);
        oskar_mem_random_range(oskar_sky_I(sky), 1.0, 2.0, &status);
        oskar_mem_random_range(oskar_sky_Q(sky), 0.1, 1.0, &status);
        oskar_mem_random_range(oskar_sky_U(sky), 0.1, 0.5, &status);
        oskar_mem_random_range(oskar_sky_V(sky), 0.1, 0.2, &status);
        oskar_mem_random_range(oskar_sky_l(sky), 0.1, 0.9, &status);
        oskar_mem_random_range(oskar_sky_m(sky), 0.1, 0.9, &status);
        oskar_mem_random_range(oskar_sky_n(sky), 0.1, 0.9, &status);
        oskar_mem_random_range(oskar_sky_gaussian_a(sky), 0.1e-6, 0.2e-6,
                &status);
        oskar_mem_random_range(oskar_sky_gaussian_b(sky), 0.1e-6, 0.2e-6,
                &status);
        oskar_mem_random_range(oskar_sky_gaussian_c(sky), 0.1e-6, 0.2e-6,
                &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    void destroyTestData()
    {
        int status = 0;
        oskar_jones_free(jones, &status);
        oskar_mem_free(u_, &status);
        oskar_mem_free(v_, &status);
        oskar_mem_free(w_, &status);
        oskar_sky_free(sky, &status);
        oskar_telescope_free(tel, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    void runTest(int prec1, int prec2)
    {
        int num_baselines, status = 0, type;
        oskar_Mem *vis1 = 0, *vis2 = 0;
        const oskar_Mem *x, *y;
        oskar_Timer *timer1, *timer2;
        double time1 = 0.0, time2 = 0.0;
        int N = 40;
        int loc1 = OSKAR_GPU, loc2 = OSKAR_GPU, matrix = 1;

        // Create the timers.
        timer1 = oskar_timer_create(loc1 == OSKAR_GPU ?
                OSKAR_TIMER_CUDA : OSKAR_TIMER_NATIVE);
        timer2 = oskar_timer_create(loc2 == OSKAR_GPU ?
                OSKAR_TIMER_CUDA : OSKAR_TIMER_NATIVE);

        // Run first part.
        createTestData(prec1, loc1, matrix);
        x = oskar_telescope_station_true_x_enu_metres_const(tel);
        y = oskar_telescope_station_true_y_enu_metres_const(tel);
        num_baselines = oskar_telescope_num_baselines(tel);
        type = prec1 | OSKAR_COMPLEX;
        if (matrix) type |= OSKAR_MATRIX;
        vis1 = oskar_mem_create(type, loc1, num_baselines, &status);
        oskar_mem_clear_contents(vis1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        cudaDeviceSynchronize();
        oskar_timer_start(timer1);

        if (prec1 == OSKAR_SINGLE)
        {
            for (int i = 0; i < N; ++i)
            {
                oskar_correlate_point_time_smearing_cuda_f(num_sources,
                        num_stations, oskar_jones_float4c_const(jones, &status),
                        oskar_mem_float_const(oskar_sky_I_const(sky), &status),
                        oskar_mem_float_const(oskar_sky_Q_const(sky), &status),
                        oskar_mem_float_const(oskar_sky_U_const(sky), &status),
                        oskar_mem_float_const(oskar_sky_V_const(sky), &status),
                        oskar_mem_float_const(oskar_sky_l_const(sky), &status),
                        oskar_mem_float_const(oskar_sky_m_const(sky), &status),
                        oskar_mem_float_const(oskar_sky_n_const(sky), &status),
                        oskar_mem_float_const(u_, &status),
                        oskar_mem_float_const(v_, &status),
                        oskar_mem_float_const(w_, &status),
                        oskar_mem_float_const(x, &status),
                        oskar_mem_float_const(y, &status),
                        0.0, 1e10, 0.5, 1.3, 5.0, 2.0, 1.1,
                        oskar_mem_float4c(vis1, &status)
                );
            }
        }
        else
        {
            for (int i = 0; i < N; ++i)
            {
                oskar_correlate_point_time_smearing_cuda_d(num_sources,
                        num_stations, oskar_jones_double4c_const(jones, &status),
                        oskar_mem_double_const(oskar_sky_I_const(sky), &status),
                        oskar_mem_double_const(oskar_sky_Q_const(sky), &status),
                        oskar_mem_double_const(oskar_sky_U_const(sky), &status),
                        oskar_mem_double_const(oskar_sky_V_const(sky), &status),
                        oskar_mem_double_const(oskar_sky_l_const(sky), &status),
                        oskar_mem_double_const(oskar_sky_m_const(sky), &status),
                        oskar_mem_double_const(oskar_sky_n_const(sky), &status),
                        oskar_mem_double_const(u_, &status),
                        oskar_mem_double_const(v_, &status),
                        oskar_mem_double_const(w_, &status),
                        oskar_mem_double_const(x, &status),
                        oskar_mem_double_const(y, &status),
                        0.0, 1e10, 0.5, 1.3, 5.0, 2.0, 1.1,
                        oskar_mem_double4c(vis1, &status)
                );
            }
        }

        time1 = oskar_timer_elapsed(timer1) / N;
        destroyTestData();
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Run second part.
        createTestData(prec2, loc2, matrix);
        x = oskar_telescope_station_true_x_enu_metres_const(tel);
        y = oskar_telescope_station_true_y_enu_metres_const(tel);
        num_baselines = oskar_telescope_num_baselines(tel);
        type = prec2 | OSKAR_COMPLEX;
        if (matrix) type |= OSKAR_MATRIX;
        vis2 = oskar_mem_create(type, loc2, num_baselines, &status);
        oskar_mem_clear_contents(vis2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        cudaDeviceSynchronize();
        oskar_timer_start(timer2);

        if (prec2 == OSKAR_SINGLE)
        {
            for (int i = 0; i < N; ++i)
            {
                oskar_correlate_point_time_smearing_new_cuda_f(num_sources,
                        num_stations, oskar_jones_float4c_const(jones, &status),
                        oskar_mem_float_const(oskar_sky_l_const(sky), &status),
                        oskar_mem_float_const(oskar_sky_m_const(sky), &status),
                        oskar_mem_float_const(oskar_sky_n_const(sky), &status),
                        oskar_mem_float_const(u_, &status),
                        oskar_mem_float_const(v_, &status),
                        oskar_mem_float_const(w_, &status),
                        oskar_mem_float_const(x, &status),
                        oskar_mem_float_const(y, &status),
                        0.0, 1e10, 0.5, 1.3, 5.0, 2.0, 1.1,
                        oskar_mem_float4c(vis2, &status)
                );
            }
        }
        else
        {
            for (int i = 0; i < N; ++i)
            {
                oskar_correlate_point_time_smearing_new_cuda_d(num_sources,
                        num_stations, oskar_jones_double4c_const(jones, &status),
                        oskar_mem_double_const(oskar_sky_l_const(sky), &status),
                        oskar_mem_double_const(oskar_sky_m_const(sky), &status),
                        oskar_mem_double_const(oskar_sky_n_const(sky), &status),
                        oskar_mem_double_const(u_, &status),
                        oskar_mem_double_const(v_, &status),
                        oskar_mem_double_const(w_, &status),
                        oskar_mem_double_const(x, &status),
                        oskar_mem_double_const(y, &status),
                        0.0, 1e10, 0.5, 1.3, 5.0, 2.0, 1.1,
                        oskar_mem_double4c(vis2, &status)
                );
            }
        }

        time2 = oskar_timer_elapsed(timer2) / N;
        destroyTestData();
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Destroy the timers.
        oskar_timer_free(timer1);
        oskar_timer_free(timer2);

        // Free memory.
        oskar_mem_free(vis1, &status);
        oskar_mem_free(vis2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Record properties for test.
        RecordProperty("Prec1", prec1 == OSKAR_SINGLE ? "Single" : "Double");
        RecordProperty("Loc1", loc1 == OSKAR_CPU ? "CPU" : "GPU");
        RecordProperty("Time1_ms", int(time1 * 1000));
        RecordProperty("Prec2", prec2 == OSKAR_SINGLE ? "Single" : "Double");
        RecordProperty("Loc2", loc2 == OSKAR_CPU ? "CPU" : "GPU");
        RecordProperty("Time2_ms", int(time2 * 1000));

#ifdef ALLOW_PRINTING
        // Print times.
        printf("  > OLD: %.6f sec.\n", time1);
        printf("  > NEW: %.6f sec.\n", time2);
#endif
    }
};

const double correlate_new::bandwidth = 1e4;

TEST_F(correlate_new, single_precision)
{
    runTest(OSKAR_SINGLE, OSKAR_SINGLE);
}

TEST_F(correlate_new, single_precision2)
{
    runTest(OSKAR_SINGLE, OSKAR_SINGLE);
}

TEST_F(correlate_new, double_precision)
{
    runTest(OSKAR_DOUBLE, OSKAR_DOUBLE);
}

