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

#include <oskar_timer.h>

#include <oskar_auto_correlate.h>
#include <oskar_get_error_string.h>
#include <cstdlib>

// Comment out this line to disable benchmark timer printing.
 #define ALLOW_PRINTING 1

static void check_values(const oskar_Mem* approx, const oskar_Mem* accurate)
{
    int status = 0;
    double min_rel_error, max_rel_error, avg_rel_error, std_rel_error, tol;
    oskar_mem_evaluate_relative_error(approx, accurate, &min_rel_error,
            &max_rel_error, &avg_rel_error, &std_rel_error, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    tol = oskar_mem_is_double(approx) &&
            oskar_mem_is_double(accurate) ? 1e-11 : 2e-3;
    EXPECT_LT(max_rel_error, tol) << std::setprecision(5) <<
            "RELATIVE ERROR" <<
            " MIN: " << min_rel_error << " MAX: " << max_rel_error <<
            " AVG: " << avg_rel_error << " STD: " << std_rel_error;
    tol = oskar_mem_is_double(approx) &&
            oskar_mem_is_double(accurate) ? 1e-12 : 1e-5;
    EXPECT_LT(avg_rel_error, tol) << std::setprecision(5) <<
            "RELATIVE ERROR" <<
            " MIN: " << min_rel_error << " MAX: " << max_rel_error <<
            " AVG: " << avg_rel_error << " STD: " << std_rel_error;
}

class auto_correlate : public ::testing::Test
{
protected:
    static const int num_sources = 277;
    static const int num_stations = 50;
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
        sky = oskar_sky_create(precision, location, num_sources, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Fill data structures with random data in sensible ranges.
        srand(2);
        oskar_mem_random_range(oskar_jones_mem(jones), 1.0, 5.0, &status);
        oskar_mem_random_range(oskar_sky_I(sky), 1.0, 2.0, &status);
        oskar_mem_random_range(oskar_sky_Q(sky), 0.1, 1.0, &status);
        oskar_mem_random_range(oskar_sky_U(sky), 0.1, 0.5, &status);
        oskar_mem_random_range(oskar_sky_V(sky), 0.1, 0.2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    void destroyTestData()
    {
        int status = 0;
        oskar_jones_free(jones, &status);
        oskar_sky_free(sky, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    void runTest(int prec1, int prec2, int loc1, int loc2, int matrix)
    {
        int status = 0, type;
        oskar_Mem *vis1, *vis2;
        oskar_Timer *timer1, *timer2;
        double time1, time2;

        // Create the timers.
        timer1 = oskar_timer_create(loc1 == OSKAR_GPU ?
                OSKAR_TIMER_CUDA : OSKAR_TIMER_NATIVE);
        timer2 = oskar_timer_create(loc2 == OSKAR_GPU ?
                OSKAR_TIMER_CUDA : OSKAR_TIMER_NATIVE);

        // Run first part.
        createTestData(prec1, loc1, matrix);
        type = prec1 | OSKAR_COMPLEX;
        if (matrix) type |= OSKAR_MATRIX;
        vis1 = oskar_mem_create(type, loc1, num_stations, &status);
        oskar_mem_clear_contents(vis1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_timer_start(timer1);
        oskar_auto_correlate(vis1, oskar_sky_num_sources(sky), jones, sky,
                &status);
        time1 = oskar_timer_elapsed(timer1);
        destroyTestData();
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Run second part.
        createTestData(prec2, loc2, matrix);
        type = prec2 | OSKAR_COMPLEX;
        if (matrix) type |= OSKAR_MATRIX;
        vis2 = oskar_mem_create(type, loc2, num_stations, &status);
        oskar_mem_clear_contents(vis2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_timer_start(timer2);
        oskar_auto_correlate(vis2, oskar_sky_num_sources(sky), jones, sky,
                &status);
        time2 = oskar_timer_elapsed(timer2);
        destroyTestData();
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Destroy the timers.
        oskar_timer_free(timer1);
        oskar_timer_free(timer2);

        // Compare results.
        check_values(vis1, vis2);

        // Free memory.
        oskar_mem_free(vis1, &status);
        oskar_mem_free(vis2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Record properties for test.
        RecordProperty("JonesType", matrix ? "Matrix" : "Scalar");
        RecordProperty("Prec1", prec1 == OSKAR_SINGLE ? "Single" : "Double");
        RecordProperty("Loc1", loc1 == OSKAR_CPU ? "CPU" : "GPU");
        RecordProperty("Time1_ms", int(time1 * 1000));
        RecordProperty("Prec2", prec2 == OSKAR_SINGLE ? "Single" : "Double");
        RecordProperty("Loc2", loc2 == OSKAR_CPU ? "CPU" : "GPU");
        RecordProperty("Time2_ms", int(time2 * 1000));

#ifdef ALLOW_PRINTING
        // Print times.
        printf("  > %s.\n", matrix ? "Matrix" : "Scalar");
        printf("    %s precision %s: %.2f ms, %s precision %s: %.2f ms\n",
                prec1 == OSKAR_SINGLE ? "Single" : "Double",
                loc1 == OSKAR_CPU ? "CPU" : "GPU",
                time1 * 1000.0,
                prec2 == OSKAR_SINGLE ? "Single" : "Double",
                loc2 == OSKAR_CPU ? "CPU" : "GPU",
                time2 * 1000.0);
#endif
    }
};

// CPU only.
TEST_F(auto_correlate, matrix_singleCPU_doubleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_CPU, OSKAR_CPU, 1);
}

#ifdef OSKAR_HAVE_CUDA
TEST_F(auto_correlate, matrix_singleGPU_doubleGPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_GPU, 1);
}

TEST_F(auto_correlate, matrix_singleGPU_singleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_SINGLE,
            OSKAR_GPU, OSKAR_CPU, 1);
}

TEST_F(auto_correlate, matrix_doubleGPU_doubleCPU)
{
    runTest(OSKAR_DOUBLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_CPU, 1);
}

TEST_F(auto_correlate, matrix_singleGPU_doubleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_CPU, 1);
}

TEST_F(auto_correlate, matrix_singleCPU_doubleGPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_CPU, OSKAR_GPU, 1);
}
#endif

// SCALAR VERSIONS ////////////////////////////////////////////////////////////

// CPU only.
TEST_F(auto_correlate, scalar_singleCPU_doubleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_CPU, OSKAR_CPU, 0);
}

#ifdef OSKAR_HAVE_CUDA
TEST_F(auto_correlate, scalar_singleGPU_doubleGPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_GPU, 0);
}

TEST_F(auto_correlate, scalar_singleGPU_singleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_SINGLE,
            OSKAR_GPU, OSKAR_CPU, 0);
}

TEST_F(auto_correlate, scalar_doubleGPU_doubleCPU)
{
    runTest(OSKAR_DOUBLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_CPU, 0);
}

TEST_F(auto_correlate, scalar_singleGPU_doubleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_CPU, 0);
}

TEST_F(auto_correlate, scalar_singleCPU_doubleGPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_CPU, OSKAR_GPU, 0);
}
#endif
