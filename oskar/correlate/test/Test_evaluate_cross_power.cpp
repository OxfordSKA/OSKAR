/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "utility/oskar_timer.h"

#include "correlate/oskar_evaluate_cross_power.h"
#include "utility/oskar_get_error_string.h"
#include <cstdlib>

// Comment out this line to disable benchmark timer printing.
 #define ALLOW_PRINTING 1

static void check_values(const oskar_Mem* approx, const oskar_Mem* accurate)
{
    int status = 0;
    double min_rel_error = 0.0, max_rel_error = 0.0;
    double avg_rel_error = 0.0, std_rel_error = 0.0, tol = 0.0;
    oskar_mem_evaluate_relative_error(approx, accurate, &min_rel_error,
            &max_rel_error, &avg_rel_error, &std_rel_error, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    tol = oskar_mem_is_double(approx) &&
            oskar_mem_is_double(accurate) ? 1e-11 : 1e-4;
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

class cross_power : public ::testing::Test
{
protected:
    static const int num_sources = 277;
    static const int num_stations = 7;
    oskar_Mem* jones;

protected:
    void createTestData(int precision, int location, int matrix)
    {
        int status = 0, type = 0;

        // Allocate memory for data structures.
        type = precision | OSKAR_COMPLEX;
        if (matrix) type |= OSKAR_MATRIX;
        jones = oskar_mem_create(type, location, num_stations * num_sources,
                &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Fill data structures with random data in sensible ranges.
        srand(0);
        oskar_mem_random_range(jones, 1.0, 10.0, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    void destroyTestData()
    {
        int status = 0;
        oskar_mem_free(jones, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    void runTest(int prec1, int prec2, int loc1, int loc2, int matrix)
    {
        int status = 0, type = 0;
        oskar_Mem *beam1 = 0, *beam2 = 0;
        oskar_Timer *timer1 = 0, *timer2 = 0;
        double time1 = 0.0, time2 = 0.0;

        // Create the timers.
        timer1 = oskar_timer_create(loc1 == OSKAR_GPU ?
                OSKAR_TIMER_CUDA : OSKAR_TIMER_NATIVE);
        timer2 = oskar_timer_create(loc2 == OSKAR_GPU ?
                OSKAR_TIMER_CUDA : OSKAR_TIMER_NATIVE);

        // Run first part.
        type = prec1 | OSKAR_COMPLEX;
        if (matrix) type |= OSKAR_MATRIX;
        beam1 = oskar_mem_create(type, loc1, num_sources, &status);
        oskar_mem_clear_contents(beam1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        createTestData(prec1, loc1, matrix);
        oskar_timer_start(timer1);
        oskar_evaluate_cross_power(num_sources, num_stations, jones,
                1.0, 0.0, 0.0, 0.0, 0, beam1, &status);
        time1 = oskar_timer_elapsed(timer1);
        destroyTestData();
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Run second part.
        type = prec2 | OSKAR_COMPLEX;
        if (matrix) type |= OSKAR_MATRIX;
        beam2 = oskar_mem_create(type, loc2, num_sources, &status);
        oskar_mem_clear_contents(beam2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        createTestData(prec2, loc2, matrix);
        oskar_timer_start(timer2);
        oskar_evaluate_cross_power(num_sources, num_stations, jones,
                1.0, 0.0, 0.0, 0.0, 0, beam2, &status);
        time2 = oskar_timer_elapsed(timer2);
        destroyTestData();
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Destroy the timers.
        oskar_timer_free(timer1);
        oskar_timer_free(timer2);

        // Compare results.
        check_values(beam1, beam2);

        // Free memory.
        oskar_mem_free(beam1, &status);
        oskar_mem_free(beam2, &status);
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
TEST_F(cross_power, matrix_singleCPU_doubleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_CPU, OSKAR_CPU, 1);
}

#ifdef OSKAR_HAVE_CUDA
TEST_F(cross_power, matrix_singleGPU_doubleGPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_GPU, 1);
}

TEST_F(cross_power, matrix_singleGPU_singleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_SINGLE,
            OSKAR_GPU, OSKAR_CPU, 1);
}

TEST_F(cross_power, matrix_doubleGPU_doubleCPU)
{
    runTest(OSKAR_DOUBLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_CPU, 1);
}

TEST_F(cross_power, matrix_singleGPU_doubleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_CPU, 1);
}

TEST_F(cross_power, matrix_singleCPU_doubleGPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_CPU, OSKAR_GPU, 1);
}
#endif

// SCALAR VERSIONS.

// CPU only.
TEST_F(cross_power, scalar_singleCPU_doubleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_CPU, OSKAR_CPU, 0);
}

#ifdef OSKAR_HAVE_CUDA
TEST_F(cross_power, scalar_singleGPU_doubleGPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_GPU, 0);
}

TEST_F(cross_power, scalar_singleGPU_singleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_SINGLE,
            OSKAR_GPU, OSKAR_CPU, 0);
}

TEST_F(cross_power, scalar_doubleGPU_doubleCPU)
{
    runTest(OSKAR_DOUBLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_CPU, 0);
}

TEST_F(cross_power, scalar_singleGPU_doubleCPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_CPU, 0);
}

TEST_F(cross_power, scalar_singleCPU_doubleGPU)
{
    runTest(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_CPU, OSKAR_GPU, 0);
}
#endif
