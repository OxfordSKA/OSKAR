/*
 * Copyright (c) 2013, The University of Oxford
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

#include <oskar_timer_functions.h>

#include <oskar_correlate.h>
#include <oskar_get_error_string.h>
#include <oskar_jones_init.h>
#include <oskar_jones_free.h>
#include <oskar_mem_clear_contents.h>
#include <oskar_mem_copy.h>
#include <oskar_mem_element_multiply.h>
#include <oskar_mem_evaluate_relative_error.h>
#include <oskar_mem_free.h>
#include <oskar_mem_init.h>
#include <oskar_mem_random_fill.h>
#include <oskar_mem_type_check.h>
#include <oskar_sky_model_init.h>
#include <oskar_sky_model_free.h>
#include <oskar_telescope_model_init.h>
#include <oskar_telescope_model_free.h>
#include <oskar_kahan_sum.h>
#include <cstdlib>

#define TOL_FLT 1e-6
#define TOL_DBL 1e-12

// Comment out this line to disable benchmark timer printing.
// #define ALLOW_PRINTING 1

static void check_values(const oskar_Mem* approx, const oskar_Mem* accurate)
{
    int status = 0;
    double min_rel_error, max_rel_error, avg_rel_error, std_rel_error, tol;
    oskar_mem_evaluate_relative_error(approx, accurate, &min_rel_error,
            &max_rel_error, &avg_rel_error, &std_rel_error, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    tol = oskar_mem_is_double(approx->type) &&
            oskar_mem_is_double(accurate->type) ? TOL_DBL : TOL_FLT;
    EXPECT_LT(max_rel_error, tol) << std::setprecision(5) <<
            "RELATIVE ERROR" <<
            " MIN: " << min_rel_error << " MAX: " << max_rel_error <<
            " AVG: " << avg_rel_error << " STD: " << std_rel_error;
    EXPECT_LT(avg_rel_error, tol) << std::setprecision(5) <<
            "RELATIVE ERROR" <<
            " MIN: " << min_rel_error << " MAX: " << max_rel_error <<
            " AVG: " << avg_rel_error << " STD: " << std_rel_error;
}

class correlate : public ::testing::Test
{
protected:
    static const int num_sources = 10000;
    static const int num_stations = 30;
    static const int num_baselines = num_stations * (num_stations - 1) / 2;
    oskar_Mem u_, v_;
    oskar_TelescopeModel telescope_;
    oskar_SkyModel sky_;
    oskar_Jones jones_;

protected:
    void createTestData(int precision, int location)
    {
        int status = 0;

        // Allocate memory for data structures.
        oskar_jones_init(&jones_, precision | OSKAR_COMPLEX | OSKAR_MATRIX,
                location, num_stations, num_sources, &status);
        oskar_mem_init(&u_, precision, location, num_stations, 1, &status);
        oskar_mem_init(&v_, precision, location, num_stations, 1, &status);
        oskar_sky_model_init(&sky_, precision, location, num_sources, &status);
        oskar_telescope_model_init(&telescope_, precision, location,
                num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Fill data structures with random data in sensible ranges.
        srand(0);
        oskar_mem_random_fill(&jones_.data, 0.1, 100.0, &status);
        oskar_mem_random_fill(&u_, 500.0, 1000.0, &status);
        oskar_mem_random_fill(&v_, 500.0, 1000.0, &status);
        oskar_mem_random_fill(&telescope_.station_x, 0.1, 1000.0, &status);
        oskar_mem_random_fill(&telescope_.station_y, 0.1, 1000.0, &status);
        oskar_mem_random_fill(&telescope_.station_z, 0.1, 1000.0, &status);
        oskar_mem_random_fill(&sky_.I, 2.0, 5.0, &status);
        oskar_mem_random_fill(&sky_.Q, 0.1, 1.0, &status);
        oskar_mem_random_fill(&sky_.U, 0.1, 0.5, &status);
        oskar_mem_random_fill(&sky_.V, 0.1, 0.2, &status);
        oskar_mem_random_fill(&sky_.l, 0.1, 0.9, &status);
        oskar_mem_random_fill(&sky_.m, 0.1, 0.9, &status);
        oskar_mem_random_fill(&sky_.n, 0.1, 0.9, &status);
        oskar_mem_random_fill(&sky_.gaussian_a, 0.1e-6, 0.2e-6, &status);
        oskar_mem_random_fill(&sky_.gaussian_b, 0.1e-6, 0.2e-6, &status);
        oskar_mem_random_fill(&sky_.gaussian_c, 0.1e-6, 0.2e-6, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Set wavelength and bandwidth.
        telescope_.wavelength_metres = 1.0;
        telescope_.bandwidth_hz = 10000.0;
    }

    void destroyTestData()
    {
        int status = 0;
        oskar_jones_free(&jones_, &status);
        oskar_mem_free(&u_, &status);
        oskar_mem_free(&v_, &status);
        oskar_sky_model_free(&sky_, &status);
        oskar_telescope_model_free(&telescope_, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    void runTest(int prec1, int prec2, int loc1, int loc2, int extended,
            double time_average)
    {
        int status = 0, type;
        oskar_Mem vis1, vis2;
        oskar_Timer timer1, timer2;
        double time1, time2;

        // Create the timers.
        oskar_timer_create(&timer1, loc1 == OSKAR_LOCATION_GPU ?
                OSKAR_TIMER_CUDA : OSKAR_TIMER_NATIVE);
        oskar_timer_create(&timer2, loc2 == OSKAR_LOCATION_GPU ?
                OSKAR_TIMER_CUDA : OSKAR_TIMER_NATIVE);

        // Run first part.
        type = prec1 | OSKAR_COMPLEX | OSKAR_MATRIX;
        oskar_mem_init(&vis1, type, loc1, num_baselines, 1, &status);
        oskar_mem_clear_contents(&vis1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        createTestData(prec1, loc1);
        sky_.use_extended = extended;
        telescope_.time_average_sec = time_average;
        oskar_timer_start(&timer1);
        oskar_correlate(&vis1, &jones_, &telescope_, &sky_, &u_, &v_, 1.0,
                &status);
        time1 = oskar_timer_elapsed(&timer1);
        destroyTestData();
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Run second part.
        type = prec2 | OSKAR_COMPLEX | OSKAR_MATRIX;
        oskar_mem_init(&vis2, type, loc2, num_baselines, 1, &status);
        oskar_mem_clear_contents(&vis2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        createTestData(prec2, loc2);
        sky_.use_extended = extended;
        telescope_.time_average_sec = time_average;
        oskar_timer_start(&timer2);
        oskar_correlate(&vis2, &jones_, &telescope_, &sky_, &u_, &v_, 1.0,
                &status);
        time2 = oskar_timer_elapsed(&timer2);
        destroyTestData();
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Destroy the timers.
        oskar_timer_destroy(&timer1);
        oskar_timer_destroy(&timer2);

        // Compare results.
        check_values(&vis1, &vis2);

        // Free memory.
        oskar_mem_free(&vis1, &status);
        oskar_mem_free(&vis2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Record properties for test.
        RecordProperty("SourceType", extended ? "Gaussian" : "Point");
        RecordProperty("TimeSmearing", time_average == 0.0 ? "off" : "on");
        RecordProperty("Prec1", prec1 == OSKAR_SINGLE ? "Single" : "Double");
        RecordProperty("Loc1", loc1 == OSKAR_LOCATION_CPU ? "CPU" : "GPU");
        RecordProperty("Time1_ms", int(time1 * 1000));
        RecordProperty("Prec2", prec2 == OSKAR_SINGLE ? "Single" : "Double");
        RecordProperty("Loc2", loc2 == OSKAR_LOCATION_CPU ? "CPU" : "GPU");
        RecordProperty("Time2_ms", int(time2 * 1000));

#ifdef ALLOW_PRINTING
        // Print times.
        printf("  > %s sources. Time smearing %s.\n",
                extended ? "Gaussian" : "Point",
                time_average == 0.0 ? "off" : "on");
        printf("    %s precision %s: %.2f ms, %s precision %s: %.2f ms\n",
                prec1 == OSKAR_SINGLE ? "Single" : "Double",
                loc1 == OSKAR_LOCATION_CPU ? "CPU" : "GPU",
                time1 * 1000.0,
                prec2 == OSKAR_SINGLE ? "Single" : "Double",
                loc2 == OSKAR_LOCATION_CPU ? "CPU" : "GPU",
                time2 * 1000.0);
#endif
    }
};

// CPU only.
TEST_F(correlate, point_singleCPU_doubleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_CPU, OSKAR_LOCATION_CPU, 0, 0.0);
}

#ifdef OSKAR_HAVE_CUDA
TEST_F(correlate, point_singleGPU_doubleGPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_GPU, OSKAR_LOCATION_GPU, 0, 0.0);
}

TEST_F(correlate, point_singleGPU_singleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_SINGLE,
            OSKAR_LOCATION_GPU, OSKAR_LOCATION_CPU, 0, 0.0);
}

TEST_F(correlate, point_doubleGPU_doubleCPU)
{
    runTest(OSKAR_DOUBLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_GPU, OSKAR_LOCATION_CPU, 0, 0.0);
}

TEST_F(correlate, point_singleGPU_doubleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_GPU, OSKAR_LOCATION_CPU, 0, 0.0);
}

TEST_F(correlate, point_singleCPU_doubleGPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_CPU, OSKAR_LOCATION_GPU, 0, 0.0);
}
#endif

// CPU only.
TEST_F(correlate, point_timeSmearing_singleCPU_doubleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_CPU, OSKAR_LOCATION_CPU, 0, 10.0);
}

#ifdef OSKAR_HAVE_CUDA
TEST_F(correlate, point_timeSmearing_singleGPU_doubleGPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_GPU, OSKAR_LOCATION_GPU, 0, 10.0);
}

TEST_F(correlate, point_timeSmearing_singleGPU_singleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_SINGLE,
            OSKAR_LOCATION_GPU, OSKAR_LOCATION_CPU, 0, 10.0);
}

TEST_F(correlate, point_timeSmearing_doubleGPU_doubleCPU)
{
    runTest(OSKAR_DOUBLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_GPU, OSKAR_LOCATION_CPU, 0, 10.0);
}

TEST_F(correlate, point_timeSmearing_singleGPU_doubleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_GPU, OSKAR_LOCATION_CPU, 0, 10.0);
}

TEST_F(correlate, point_timeSmearing_singleCPU_doubleGPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_CPU, OSKAR_LOCATION_GPU, 0, 10.0);
}
#endif

// CPU only.
TEST_F(correlate, gaussian_singleCPU_doubleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_CPU, OSKAR_LOCATION_CPU, 1, 0.0);
}

#ifdef OSKAR_HAVE_CUDA
TEST_F(correlate, gaussian_singleGPU_doubleGPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_GPU, OSKAR_LOCATION_GPU, 1, 0.0);
}

TEST_F(correlate, gaussian_singleGPU_singleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_SINGLE,
            OSKAR_LOCATION_GPU, OSKAR_LOCATION_CPU, 1, 0.0);
}

TEST_F(correlate, gaussian_doubleGPU_doubleCPU)
{
    runTest(OSKAR_DOUBLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_GPU, OSKAR_LOCATION_CPU, 1, 0.0);
}

TEST_F(correlate, gaussian_singleGPU_doubleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_GPU, OSKAR_LOCATION_CPU, 1, 0.0);
}

TEST_F(correlate, gaussian_singleCPU_doubleGPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_CPU, OSKAR_LOCATION_GPU, 1, 0.0);
}
#endif

// CPU only.
TEST_F(correlate, gaussian_timeSmearing_singleCPU_doubleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_CPU, OSKAR_LOCATION_CPU, 1, 10.0);
}

#ifdef OSKAR_HAVE_CUDA
TEST_F(correlate, gaussian_timeSmearing_singleGPU_doubleGPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_GPU, OSKAR_LOCATION_GPU, 1, 10.0);
}

TEST_F(correlate, gaussian_timeSmearing_singleGPU_singleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_SINGLE,
            OSKAR_LOCATION_GPU, OSKAR_LOCATION_CPU, 1, 10.0);
}

TEST_F(correlate, gaussian_timeSmearing_doubleGPU_doubleCPU)
{
    runTest(OSKAR_DOUBLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_GPU, OSKAR_LOCATION_CPU, 1, 10.0);
}

TEST_F(correlate, gaussian_timeSmearing_singleGPU_doubleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_GPU, OSKAR_LOCATION_CPU, 1, 10.0);
}

TEST_F(correlate, gaussian_timeSmearing_singleCPU_doubleGPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_LOCATION_CPU, OSKAR_LOCATION_GPU, 1, 10.0);
}
#endif

#if 0
TEST(KahanSum, sum)
{
    float sum_normal = 0.0f;
    int num = 100000;
    srand(0);
    for (int i = 0; i < num; ++i)
    {
        float r = (float)rand() / (float)RAND_MAX;
        sum_normal += r;
    }
    printf("Sum (normal, single): %.6f\n", sum_normal);

    float sum_kahan = 0.0f, kahan_guard = 0.0f;
    srand(0);
    for (int i = 0; i < num; ++i)
    {
        float r = (float)rand() / (float)RAND_MAX;
        oskar_kahan_sum_f(&sum_kahan, r, &kahan_guard);
    }
    printf("Sum (Kahan, single) : %.6f\n", sum_kahan);

    double sum_normal_double = 0.0;
    srand(0);
    for (int i = 0; i < num; ++i)
    {
        double r = (double)rand() / (double)RAND_MAX;
        sum_normal_double += r;
    }
    printf("Sum (normal, double): %.6f\n", sum_normal_double);
}
#endif
