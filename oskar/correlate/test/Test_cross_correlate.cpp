/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "utility/oskar_timer.h"

#include "correlate/oskar_cross_correlate.h"
#include "utility/oskar_get_error_string.h"
#include "math/oskar_kahan_sum.h"
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
            oskar_mem_is_double(accurate) ? 1e-10 : 2e-2;
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

static const char* loc_to_str(int loc)
{
    switch (loc) {
    case OSKAR_CPU: return "CPU";
    case OSKAR_GPU: return "GPU";
    case OSKAR_CL:  return "CL";
    default: return "";
    }
}

class cross_correlate : public ::testing::Test
{
protected:
    static const int num_sources = 277;
    static const int num_stations = 19;
    oskar_Mem *src_dir[3], *src_ext[3], *src_flux[4], *uvw[3];
    oskar_Telescope* tel;
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
        for (int i = 0; i < 3; ++i)
        {
            src_dir[i] = oskar_mem_create(
                    precision, location, num_sources, &status);
            src_ext[i] = oskar_mem_create(
                    precision, location, num_sources, &status);
            uvw[i] = oskar_mem_create(
                    precision, location, num_stations, &status);
        }
        for (int i = 0; i < 4; ++i)
        {
            src_flux[i] = oskar_mem_create(
                    precision, location, num_sources, &status);
        }
        tel = oskar_telescope_create(precision, location,
                num_stations, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Fill data structures with random data in sensible ranges.
#ifdef _MSC_VER
        srand(1);
#else
        srand(2);
#endif
        oskar_mem_random_range(oskar_jones_mem(jones), 1.0, 5.0, &status);
        oskar_mem_random_range(uvw[0], 1.0, 5.0, &status);
        oskar_mem_random_range(uvw[1], 1.0, 5.0, &status);
        oskar_mem_random_range(uvw[2], 1.0, 5.0, &status);
        oskar_mem_random_range(
                oskar_telescope_station_true_offset_ecef_metres(tel, 0),
                0.1, 1000.0, &status);
        oskar_mem_random_range(
                oskar_telescope_station_true_offset_ecef_metres(tel, 1),
                0.1, 1000.0, &status);
        oskar_mem_random_range(
                oskar_telescope_station_true_offset_ecef_metres(tel, 2),
                0.1, 1000.0, &status);
        oskar_mem_random_range(src_flux[0], 1.0, 2.0, &status);
        oskar_mem_random_range(src_flux[1], 0.1, 1.0, &status);
        oskar_mem_random_range(src_flux[2], 0.1, 0.5, &status);
        oskar_mem_random_range(src_flux[3], 0.1, 0.2, &status);
        oskar_mem_random_range(src_dir[0], 0.1, 0.9, &status);
        oskar_mem_random_range(src_dir[1], 0.1, 0.9, &status);
        oskar_mem_random_range(src_dir[2], 0.1, 0.9, &status);
        oskar_mem_random_range(src_ext[0], 0.1e-6, 0.2e-6, &status);
        oskar_mem_random_range(src_ext[1], 0.1e-6, 0.2e-6, &status);
        oskar_mem_random_range(src_ext[2], 0.1e-6, 0.2e-6, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    void destroy_test_data()
    {
        int status = 0;
        oskar_jones_free(jones, &status);
        for (int i = 0; i < 3; ++i)
        {
            oskar_mem_free(src_dir[i], &status);
            oskar_mem_free(src_ext[i], &status);
            oskar_mem_free(uvw[i], &status);
        }
        for (int i = 0; i < 4; ++i)
        {
            oskar_mem_free(src_flux[i], &status);
        }
        oskar_telescope_free(tel, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
    }

    void run_test(int prec1, int prec2, int loc1, int loc2, int matrix,
            int extended, double time_average, double freq_average)
    {
        int num_baselines = 0, status = 0, type = 0;
        oskar_Mem *vis1 = 0, *vis2 = 0;
        oskar_Timer *timer1 = 0, *timer2 = 0;
        double time1 = 0.0, time2 = 0.0, frequency = 100e6;

        // Create the timers.
        timer1 = oskar_timer_create(loc1);
        timer2 = oskar_timer_create(loc2);

        // Run first part.
        create_test_data(prec1, loc1, matrix);
        num_baselines = oskar_telescope_num_baselines(tel);
        type = prec1 | OSKAR_COMPLEX;
        if (matrix) type |= OSKAR_MATRIX;
        vis1 = oskar_mem_create(type, loc1, num_baselines, &status);
        oskar_mem_clear_contents(vis1, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_telescope_set_channel_bandwidth(tel, freq_average);
        oskar_telescope_set_time_average(tel, time_average);
        oskar_timer_start(timer1);
        oskar_cross_correlate(extended, num_sources, jones,
                src_flux, src_dir, src_ext,
                tel, uvw, 1.0, frequency, 0, vis1, &status);
        time1 = oskar_timer_elapsed(timer1);
        destroy_test_data();
        ASSERT_EQ(0, status) << oskar_get_error_string(status);

        // Run second part.
        create_test_data(prec2, loc2, matrix);
        num_baselines = oskar_telescope_num_baselines(tel);
        type = prec2 | OSKAR_COMPLEX;
        if (matrix) type |= OSKAR_MATRIX;
        vis2 = oskar_mem_create(type, loc2, num_baselines, &status);
        oskar_mem_clear_contents(vis2, &status);
        ASSERT_EQ(0, status) << oskar_get_error_string(status);
        oskar_telescope_set_channel_bandwidth(tel, freq_average);
        oskar_telescope_set_time_average(tel, time_average);
        oskar_timer_start(timer2);
        oskar_cross_correlate(extended, num_sources, jones,
                src_flux, src_dir, src_ext,
                tel, uvw, 1.0, frequency, 0, vis2, &status);
        time2 = oskar_timer_elapsed(timer2);
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
        RecordProperty("SourceType", extended ? "Gaussian" : "Point");
        RecordProperty("JonesType", matrix ? "Matrix" : "Scalar");
        RecordProperty("TimeSmearing", time_average == 0.0 ? "off" : "on");
        RecordProperty("Prec1", prec1 == OSKAR_SINGLE ? "Single" : "Double");
        RecordProperty("Loc1", loc_to_str(loc1));
        RecordProperty("Time1_ms", int(time1 * 1000));
        RecordProperty("Prec2", prec2 == OSKAR_SINGLE ? "Single" : "Double");
        RecordProperty("Loc2", loc_to_str(loc2));
        RecordProperty("Time2_ms", int(time2 * 1000));

#ifdef ALLOW_PRINTING
        // Print times.
        printf("  > %s. %s sources. Time smearing %s. Bandwidth smearing %s.\n",
                matrix ? "Matrix" : "Scalar",
                extended ? "Gaussian" : "Point",
                time_average == 0.0 ? "off" : "on",
                freq_average == 0.0 ? "off" : "on");
        printf("    %s precision %s: %.2f ms, %s precision %s: %.2f ms\n",
                prec1 == OSKAR_SINGLE ? "Single" : "Double",
                loc_to_str(loc1),
                time1 * 1000.0,
                prec2 == OSKAR_SINGLE ? "Single" : "Double",
                loc_to_str(loc2),
                time2 * 1000.0);
#endif
    }
};


// CPU only.
// Check consistency between single and double precision.
TEST_F(cross_correlate, CPU)
{
    const int matrix_type[] = {0, 1};
    const int source_type[] = {0, 1};
    const double time_avg[] = {0.0, 10.0};
    const double freq_avg[] = {0.0, 1.e4};
    for (int i_matrix_type = 0; i_matrix_type < 2; ++i_matrix_type)
    {
        for (int i_source_type = 0; i_source_type < 2; ++i_source_type)
        {
            for (int i_time_avg = 0; i_time_avg < 2; ++i_time_avg)
            {
                for (int i_freq_avg = 0; i_freq_avg < 2; ++i_freq_avg)
                {
                    run_test(OSKAR_SINGLE, OSKAR_DOUBLE,
                            OSKAR_CPU, OSKAR_CPU,
                            matrix_type[i_matrix_type],
                            source_type[i_source_type],
                            time_avg[i_time_avg],
                            freq_avg[i_freq_avg]);
                }
            }
        }
    }
}

#ifdef OSKAR_HAVE_CUDA
// Check for consistency between CPU and CUDA versions.
TEST_F(cross_correlate, CUDA)
{
    const int precision[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    const int matrix_type[] = {0, 1};
    const int source_type[] = {0, 1};
    const double time_avg[] = {0.0, 10.0};
    const double freq_avg[] = {0.0, 1.e4};
    for (int i_prec = 0; i_prec < 2; ++i_prec)
    {
        for (int i_matrix_type = 0; i_matrix_type < 2; ++i_matrix_type)
        {
            for (int i_source_type = 0; i_source_type < 2; ++i_source_type)
            {
                for (int i_time_avg = 0; i_time_avg < 2; ++i_time_avg)
                {
                    for (int i_freq_avg = 0; i_freq_avg < 2; ++i_freq_avg)
                    {
                        run_test(precision[i_prec], precision[i_prec],
                                OSKAR_CPU, OSKAR_GPU,
                                matrix_type[i_matrix_type],
                                source_type[i_source_type],
                                time_avg[i_time_avg],
                                freq_avg[i_freq_avg]);
                    }
                }
            }
        }
    }
}
#endif

#ifdef OSKAR_HAVE_OPENCL
// Check for consistency between CPU and OpenCL versions.
TEST_F(cross_correlate, CL)
{
    const int precision[] = {OSKAR_SINGLE, OSKAR_DOUBLE};
    const int matrix_type[] = {0, 1};
    const int source_type[] = {0, 1};
    const double time_avg[] = {0.0, 10.0};
    const double freq_avg[] = {0.0, 1.e4};
    for (int i_prec = 0; i_prec < 2; ++i_prec)
    {
        for (int i_matrix_type = 0; i_matrix_type < 2; ++i_matrix_type)
        {
            for (int i_source_type = 0; i_source_type < 2; ++i_source_type)
            {
                for (int i_time_avg = 0; i_time_avg < 2; ++i_time_avg)
                {
                    for (int i_freq_avg = 0; i_freq_avg < 2; ++i_freq_avg)
                    {
                        run_test(precision[i_prec], precision[i_prec],
                                OSKAR_CPU, OSKAR_CL,
                                matrix_type[i_matrix_type],
                                source_type[i_source_type],
                                time_avg[i_time_avg],
                                freq_avg[i_freq_avg]);
                    }
                }
            }
        }
    }
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
