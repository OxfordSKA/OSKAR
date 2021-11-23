/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "utility/oskar_timer.h"

#include "correlate/oskar_auto_correlate.h"
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
    oskar_Mem* src_flux[4];
    oskar_Jones* jones;

protected:
    void create_test_data(int precision, int location, int matrix)
    {
        int status = 0, type = 0;

        // Allocate memory for data structures.
        type = precision | OSKAR_COMPLEX;
        if (matrix) type |= OSKAR_MATRIX;
        jones = oskar_jones_create(type, location, num_stations, num_sources,
                &status);
        for (int i = 0; i < 4; ++i)
        {
            src_flux[i] = oskar_mem_create(
                    precision, location, num_sources, &status);
        }
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Fill data structures with random data in sensible ranges.
        srand(2);
        oskar_mem_random_range(oskar_jones_mem(jones), 1.0, 5.0, &status);
        oskar_mem_random_range(src_flux[0], 1.0, 2.0, &status);
        oskar_mem_random_range(src_flux[1], 0.1, 1.0, &status);
        oskar_mem_random_range(src_flux[2], 0.1, 0.5, &status);
        oskar_mem_random_range(src_flux[3], 0.1, 0.2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    void destroy_test_data()
    {
        int status = 0;
        oskar_jones_free(jones, &status);
        for (int i = 0; i < 4; ++i)
        {
            oskar_mem_free(src_flux[i], &status);
        }
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    void run_test(int prec1, int prec2, int loc1, int loc2, int matrix)
    {
        int status = 0, type = 0;
        oskar_Mem *vis1 = 0, *vis2 = 0;
        oskar_Timer *timer1 = 0, *timer2 = 0;

        // Create the timers.
        timer1 = oskar_timer_create(loc1 == OSKAR_GPU ?
                OSKAR_TIMER_CUDA : OSKAR_TIMER_NATIVE);
        timer2 = oskar_timer_create(loc2 == OSKAR_GPU ?
                OSKAR_TIMER_CUDA : OSKAR_TIMER_NATIVE);

        // Run first part.
        create_test_data(prec1, loc1, matrix);
        type = prec1 | OSKAR_COMPLEX;
        if (matrix) type |= OSKAR_MATRIX;
        vis1 = oskar_mem_create(type, loc1, num_stations, &status);
        oskar_mem_clear_contents(vis1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_timer_start(timer1);
        oskar_auto_correlate(num_sources, jones, src_flux, 0, vis1, &status);
        const double time1 = oskar_timer_elapsed(timer1);
        destroy_test_data();
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Run second part.
        create_test_data(prec2, loc2, matrix);
        type = prec2 | OSKAR_COMPLEX;
        if (matrix) type |= OSKAR_MATRIX;
        vis2 = oskar_mem_create(type, loc2, num_stations, &status);
        oskar_mem_clear_contents(vis2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_timer_start(timer2);
        oskar_auto_correlate(num_sources, jones, src_flux, 0, vis2, &status);
        const double time2 = oskar_timer_elapsed(timer2);
        destroy_test_data();
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
    run_test(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_CPU, OSKAR_CPU, 1);
}

#ifdef OSKAR_HAVE_CUDA
TEST_F(auto_correlate, matrix_singleGPU_doubleGPU)
{
    run_test(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_GPU, 1);
}

TEST_F(auto_correlate, matrix_singleGPU_singleCPU)
{
    run_test(OSKAR_SINGLE, OSKAR_SINGLE,
            OSKAR_GPU, OSKAR_CPU, 1);
}

TEST_F(auto_correlate, matrix_doubleGPU_doubleCPU)
{
    run_test(OSKAR_DOUBLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_CPU, 1);
}

TEST_F(auto_correlate, matrix_singleGPU_doubleCPU)
{
    run_test(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_CPU, 1);
}

TEST_F(auto_correlate, matrix_singleCPU_doubleGPU)
{
    run_test(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_CPU, OSKAR_GPU, 1);
}
#endif

// SCALAR VERSIONS ////////////////////////////////////////////////////////////

// CPU only.
TEST_F(auto_correlate, scalar_singleCPU_doubleCPU)
{
    run_test(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_CPU, OSKAR_CPU, 0);
}

#ifdef OSKAR_HAVE_CUDA
TEST_F(auto_correlate, scalar_singleGPU_doubleGPU)
{
    run_test(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_GPU, 0);
}

TEST_F(auto_correlate, scalar_singleGPU_singleCPU)
{
    run_test(OSKAR_SINGLE, OSKAR_SINGLE,
            OSKAR_GPU, OSKAR_CPU, 0);
}

TEST_F(auto_correlate, scalar_doubleGPU_doubleCPU)
{
    run_test(OSKAR_DOUBLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_CPU, 0);
}

TEST_F(auto_correlate, scalar_singleGPU_doubleCPU)
{
    run_test(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_GPU, OSKAR_CPU, 0);
}

TEST_F(auto_correlate, scalar_singleCPU_doubleGPU)
{
    run_test(OSKAR_SINGLE, OSKAR_DOUBLE,
            OSKAR_CPU, OSKAR_GPU, 0);
}
#endif
