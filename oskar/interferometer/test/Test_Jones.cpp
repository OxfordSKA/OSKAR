/*
 * Copyright (c) 2011-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "interferometer/oskar_jones.h"
#include "mem/oskar_mem.h"
#include "utility/oskar_get_error_string.h"
#include "utility/oskar_timer.h"
#include "utility/oskar_vector_types.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define CPU  OSKAR_CPU
#define GPU  OSKAR_GPU
#define CL   OSKAR_CL
#define SC   OSKAR_SINGLE_COMPLEX
#define SCM  OSKAR_SINGLE_COMPLEX_MATRIX
#define DC   OSKAR_DOUBLE_COMPLEX
#define DCM  OSKAR_DOUBLE_COMPLEX_MATRIX

static void check_values(const oskar_Mem* approx, const oskar_Mem* accurate)
{
    int status = 0;
    double min_rel_error = 0.0, max_rel_error = 0.0;
    double avg_rel_error = 0.0, std_rel_error = 0.0, tol = 0.0;
    oskar_mem_evaluate_relative_error(approx, accurate, &min_rel_error,
            &max_rel_error, &avg_rel_error, &std_rel_error, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    tol = oskar_mem_is_double(approx) &&
            oskar_mem_is_double(accurate) ? 1e-10 : 1e-4;
    EXPECT_LT(max_rel_error, tol) << std::setprecision(5) <<
            "RELATIVE ERROR" <<
            " MIN: " << min_rel_error << " MAX: " << max_rel_error <<
            " AVG: " << avg_rel_error << " STD: " << std_rel_error;
    tol = oskar_mem_is_double(approx) &&
            oskar_mem_is_double(accurate) ? 1e-12 : 1e-6;
    EXPECT_LT(avg_rel_error, tol) << std::setprecision(5) <<
            "RELATIVE ERROR" <<
            " MIN: " << min_rel_error << " MAX: " << max_rel_error <<
            " AVG: " << avg_rel_error << " STD: " << std_rel_error;
}

static const int sources = 277;
static const int stations = 19;

static void t_join(int out_typeA, int in_type1A, int in_type2A,
        int out_locA, int in_loc1A, int in_loc2A,
        int out_typeB, int in_type1B, int in_type2B,
        int out_locB, int in_loc1B, int in_loc2B,
        int expectedA, int expectedB)
{
    int status = 0;
    oskar_Timer *timerA = 0, *timerB = 0;
//    double timeA, timeB;
    oskar_Jones *in1 = 0, *in2 = 0, *outA = 0, *outB = 0;

    // Create the timers.
    timerA = oskar_timer_create(out_locA == OSKAR_GPU ?
            OSKAR_TIMER_CUDA : OSKAR_TIMER_NATIVE);
    timerB = oskar_timer_create(out_locB == OSKAR_GPU ?
            OSKAR_TIMER_CUDA : OSKAR_TIMER_NATIVE);

    // Create output blocks.
    outA = oskar_jones_create(out_typeA, out_locA, stations, sources, &status);
    outB = oskar_jones_create(out_typeB, out_locB, stations, sources, &status);

    // Run test A.
    in1 = oskar_jones_create(in_type1A, in_loc1A, stations, sources, &status);
    in2 = oskar_jones_create(in_type2A, in_loc2A, stations, sources, &status);
    srand(2);
    oskar_mem_random_range(oskar_jones_mem(in1), 1.0, 2.0, &status);
    oskar_mem_random_range(oskar_jones_mem(in2), 1.0, 2.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(timerA);
    oskar_jones_join(outA, in1, in2, &status);
//    timeA = oskar_timer_elapsed(timerA);
    EXPECT_EQ(expectedA, status);
    status = 0;
    oskar_jones_free(in1, &status);
    oskar_jones_free(in2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Run test B.
    in1 = oskar_jones_create(in_type1B, in_loc1B, stations, sources, &status);
    in2 = oskar_jones_create(in_type2B, in_loc2B, stations, sources, &status);
    srand(2);
    oskar_mem_random_range(oskar_jones_mem(in1), 1.0, 2.0, &status);
    oskar_mem_random_range(oskar_jones_mem(in2), 1.0, 2.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(timerB);
    oskar_jones_join(outB, in1, in2, &status);
//    timeB = oskar_timer_elapsed(timerB);
    EXPECT_EQ(expectedB, status);
    status = 0;
    oskar_jones_free(in1, &status);
    oskar_jones_free(in2, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Destroy the timers.
    oskar_timer_free(timerA);
    oskar_timer_free(timerB);

    // Compare results.
    check_values(oskar_jones_mem(outA), oskar_jones_mem(outB));

    // Free memory.
    oskar_jones_free(outA, &status);
    oskar_jones_free(outB, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}

static
void t_join_in_place(int in_type1A, int in_type2A, int in_loc1A, int in_loc2A,
        int in_type1B, int in_type2B, int in_loc1B, int in_loc2B,
        int expectedA, int expectedB)
{
    int status = 0;
    oskar_Timer *timerA = 0, *timerB = 0;
//    double timeA, timeB;
    oskar_Jones *in1A = 0, *in2A = 0, *in1B = 0, *in2B = 0;

    // Create the timers.
    timerA = oskar_timer_create(in_loc1A == OSKAR_GPU ?
            OSKAR_TIMER_CUDA : OSKAR_TIMER_NATIVE);
    timerB = oskar_timer_create(in_loc1B == OSKAR_GPU ?
            OSKAR_TIMER_CUDA : OSKAR_TIMER_NATIVE);

    // Run test A.
    in1A = oskar_jones_create(in_type1A, in_loc1A, stations, sources, &status);
    in2A = oskar_jones_create(in_type2A, in_loc2A, stations, sources, &status);
    srand(2);
    oskar_mem_random_range(oskar_jones_mem(in1A), 1.0, 2.0, &status);
    oskar_mem_random_range(oskar_jones_mem(in2A), 1.0, 2.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(timerA);
    oskar_jones_join(in1A, in1A, in2A, &status);
//    timeA = oskar_timer_elapsed(timerA);
    EXPECT_EQ(expectedA, status);
    status = 0;
    oskar_jones_free(in2A, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Run test B.
    in1B = oskar_jones_create(in_type1B, in_loc1B, stations, sources, &status);
    in2B = oskar_jones_create(in_type2B, in_loc2B, stations, sources, &status);
    srand(2);
    oskar_mem_random_range(oskar_jones_mem(in1B), 1.0, 2.0, &status);
    oskar_mem_random_range(oskar_jones_mem(in2B), 1.0, 2.0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(timerB);
    oskar_jones_join(in1B, in1B, in2B, &status);
//    timeB = oskar_timer_elapsed(timerB);
    EXPECT_EQ(expectedB, status);
    status = 0;
    oskar_jones_free(in2B, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Destroy the timers.
    oskar_timer_free(timerA);
    oskar_timer_free(timerB);

    // Compare results.
    check_values(oskar_jones_mem(in1A), oskar_jones_mem(in1B));

    // Free memory.
    oskar_jones_free(in1A, &status);
    oskar_jones_free(in1B, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}

static void test_ones(int precision, int location)
{
    oskar_Jones *jones = 0, *temp = 0, *j_ptr = 0;
    int status = 0, num_stations = 0, num_sources = 0;

    // Test scalar complex type.
    jones = oskar_jones_create(precision | OSKAR_COMPLEX, location,
            stations, sources, &status);
    ASSERT_EQ(0, status);
    oskar_mem_set_value_real(oskar_jones_mem(jones), 1.0,
            0, stations * sources, &status);
    EXPECT_EQ(0, status);
    j_ptr = jones;

    // Copy back to CPU memory if required.
    if (location != OSKAR_CPU)
    {
        temp = oskar_jones_create_copy(jones, OSKAR_CPU, &status);
        ASSERT_EQ(0, status);
        j_ptr = temp;
    }

    // Check data.
    num_stations = oskar_jones_num_stations(j_ptr);
    num_sources  = oskar_jones_num_sources(j_ptr);
    if (precision == OSKAR_SINGLE)
    {
        float2* p = oskar_mem_float2(oskar_jones_mem(j_ptr), &status);
        for (int st = 0; st < num_stations; ++st)
        {
            for (int src = 0, i = 0; src < num_sources; ++src, ++i)
            {
                EXPECT_FLOAT_EQ(p[i].x, 1.0f);
                EXPECT_FLOAT_EQ(p[i].y, 0.0f);
            }
        }
    }
    else
    {
        double2* p = oskar_mem_double2(oskar_jones_mem(j_ptr), &status);
        for (int st = 0; st < num_stations; ++st)
        {
            for (int src = 0, i = 0; src < num_sources; ++src, ++i)
            {
                EXPECT_DOUBLE_EQ(p[i].x, 1.0);
                EXPECT_DOUBLE_EQ(p[i].y, 0.0);
            }
        }
    }

    // Free memory.
    if (location != OSKAR_CPU)
    {
        oskar_jones_free(temp, &status);
    }
    oskar_jones_free(jones, &status);
    ASSERT_EQ(0, status);

    // Test matrix complex type.
    jones = oskar_jones_create(precision | OSKAR_COMPLEX | OSKAR_MATRIX,
            location, stations, sources, &status);
    ASSERT_EQ(0, status);
    oskar_mem_set_value_real(oskar_jones_mem(jones), 1.0,
            0, stations * sources, &status);
    EXPECT_EQ(0, status);
    j_ptr = jones;

    // Copy back to CPU memory if required.
    if (location != OSKAR_CPU)
    {
        temp = oskar_jones_create_copy(jones, OSKAR_CPU, &status);
        ASSERT_EQ(0, status);
        j_ptr = temp;
    }

    // Check data.
    num_stations = oskar_jones_num_stations(j_ptr);
    num_sources  = oskar_jones_num_sources(j_ptr);
    if (precision == OSKAR_SINGLE)
    {
        float4c* p = oskar_mem_float4c(oskar_jones_mem(j_ptr), &status);
        for (int st = 0; st < num_stations; ++st)
        {
            for (int src = 0, i = 0; src < num_sources; ++src, ++i)
            {
                EXPECT_FLOAT_EQ(p[i].a.x, 1.0f);
                EXPECT_FLOAT_EQ(p[i].a.y, 0.0f);
                EXPECT_FLOAT_EQ(p[i].b.x, 0.0f);
                EXPECT_FLOAT_EQ(p[i].b.y, 0.0f);
                EXPECT_FLOAT_EQ(p[i].c.x, 0.0f);
                EXPECT_FLOAT_EQ(p[i].c.y, 0.0f);
                EXPECT_FLOAT_EQ(p[i].d.x, 1.0f);
                EXPECT_FLOAT_EQ(p[i].d.y, 0.0f);
            }
        }
    }
    else
    {
        double4c* p = oskar_mem_double4c(oskar_jones_mem(j_ptr), &status);
        for (int st = 0; st < num_stations; ++st)
        {
            for (int src = 0, i = 0; src < num_sources; ++src, ++i)
            {
                EXPECT_DOUBLE_EQ(p[i].a.x, 1.0);
                EXPECT_DOUBLE_EQ(p[i].a.y, 0.0);
                EXPECT_DOUBLE_EQ(p[i].b.x, 0.0);
                EXPECT_DOUBLE_EQ(p[i].b.y, 0.0);
                EXPECT_DOUBLE_EQ(p[i].c.x, 0.0);
                EXPECT_DOUBLE_EQ(p[i].c.y, 0.0);
                EXPECT_DOUBLE_EQ(p[i].d.x, 1.0);
                EXPECT_DOUBLE_EQ(p[i].d.y, 0.0);
            }
        }
    }

    // Free memory.
    if (location != OSKAR_CPU)
    {
        oskar_jones_free(temp, &status);
    }
    oskar_jones_free(jones, &status);
    ASSERT_EQ(0, status);
}

// CPU only. //////////////////////////////////////////////////////////////////

TEST(Jones, join_scal_scal_scal_singleCPU_doubleCPU)
{
    t_join(SC, SC, SC, CPU, CPU, CPU,
            DC, DC, DC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_scal_singleCPU_doubleCPU)
{
    t_join(SCM, SC, SC, CPU, CPU, CPU,
            DCM, DC, DC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_matx_singleCPU_doubleCPU)
{
    t_join(SCM, SC, SCM, CPU, CPU, CPU,
            DCM, DC, DCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_matx_matx_singleCPU_doubleCPU)
{
    t_join(SCM, SCM, SCM, CPU, CPU, CPU,
            DCM, DCM, DCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_singleCPU_doubleCPU)
{
    t_join_in_place(SC, SC, CPU, CPU,
            DC, DC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_singleCPU_doubleCPU)
{
    t_join_in_place(SCM, SC, CPU, CPU,
            DCM, DC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_singleCPU_doubleCPU)
{
    t_join_in_place(SCM, SCM, CPU, CPU,
            DCM, DCM, CPU, CPU, 0, 0);
}

#ifdef OSKAR_HAVE_OPENCL

// OpenCL only. ///////////////////////////////////////////////////////////////

TEST(Jones, join_scal_scal_scal_singleCL_doubleCL)
{
    t_join(SC, SC, SC, CL, CL, CL,
            DC, DC, DC, CL, CL, CL, 0, 0);
}

TEST(Jones, join_matx_scal_scal_singleCL_doubleCL)
{
    t_join(SCM, SC, SC, CL, CL, CL,
            DCM, DC, DC, CL, CL, CL, 0, 0);
}

TEST(Jones, join_matx_scal_matx_singleCL_doubleCL)
{
    t_join(SCM, SC, SCM, CL, CL, CL,
            DCM, DC, DCM, CL, CL, CL, 0, 0);
}

TEST(Jones, join_matx_matx_matx_singleCL_doubleCL)
{
    t_join(SCM, SCM, SCM, CL, CL, CL,
            DCM, DCM, DCM, CL, CL, CL, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_singleCL_doubleCL)
{
    t_join_in_place(SC, SC, CL, CL,
            DC, DC, CL, CL, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_singleCL_doubleCL)
{
    t_join_in_place(SCM, SC, CL, CL,
            DCM, DC, CL, CL, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_singleCL_doubleCL)
{
    t_join_in_place(SCM, SCM, CL, CL,
            DCM, DCM, CL, CL, 0, 0);
}


// Compare CPU and OpenCL. ////////////////////////////////////////////////////

// Single CL, single CPU.
TEST(Jones, join_scal_scal_scal_singleCL_singleCPU)
{
    t_join(SC, SC, SC, CL, CL, CL,
            SC, SC, SC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_scal_singleCL_singleCPU)
{
    t_join(SCM, SC, SC, CL, CL, CL,
            SCM, SC, SC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_matx_singleCL_singleCPU)
{
    t_join(SCM, SC, SCM, CL, CL, CL,
            SCM, SC, SCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_matx_matx_singleCL_singleCPU)
{
    t_join(SCM, SCM, SCM, CL, CL, CL,
            SCM, SCM, SCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_singleCL_singleCPU)
{
    t_join_in_place(SC, SC, CL, CL,
            SC, SC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_singleCL_singleCPU)
{
    t_join_in_place(SCM, SC, CL, CL,
            SCM, SC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_singleCL_singleCPU)
{
    t_join_in_place(SCM, SCM, CL, CL,
            SCM, SCM, CPU, CPU, 0, 0);
}

// Double CL, double CPU.
TEST(Jones, join_scal_scal_scal_doubleCL_doubleCPU)
{
    t_join(DC, DC, DC, CL, CL, CL,
            DC, DC, DC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_scal_doubleCL_doubleCPU)
{
    t_join(DCM, DC, DC, CL, CL, CL,
            DCM, DC, DC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_matx_doubleCL_doubleCPU)
{
    t_join(DCM, DC, DCM, CL, CL, CL,
            DCM, DC, DCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_matx_matx_doubleCL_doubleCPU)
{
    t_join(DCM, DCM, DCM, CL, CL, CL,
            DCM, DCM, DCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_doubleCL_doubleCPU)
{
    t_join_in_place(DC, DC, CL, CL,
            DC, DC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_doubleCL_doubleCPU)
{
    t_join_in_place(DCM, DC, CL, CL,
            DCM, DC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_doubleCL_doubleCPU)
{
    t_join_in_place(DCM, DCM, CL, CL,
            DCM, DCM, CPU, CPU, 0, 0);
}

// Single CL, double CPU.
TEST(Jones, join_scal_scal_scal_singleCL_doubleCPU)
{
    t_join(SC, SC, SC, CL, CL, CL,
            DC, DC, DC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_scal_singleCL_doubleCPU)
{
    t_join(SCM, SC, SC, CL, CL, CL,
            DCM, DC, DC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_matx_singleCL_doubleCPU)
{
    t_join(SCM, SC, SCM, CL, CL, CL,
            DCM, DC, DCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_matx_matx_singleCL_doubleCPU)
{
    t_join(SCM, SCM, SCM, CL, CL, CL,
            DCM, DCM, DCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_singleCL_doubleCPU)
{
    t_join_in_place(SC, SC, CL, CL,
            DC, DC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_singleCL_doubleCPU)
{
    t_join_in_place(SCM, SC, CL, CL,
            DCM, DC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_singleCL_doubleCPU)
{
    t_join_in_place(SCM, SCM, CL, CL,
            DCM, DCM, CPU, CPU, 0, 0);
}

// Single CPU, double CL.
TEST(Jones, join_scal_scal_scal_singleCPU_doubleCL)
{
    t_join(SC, SC, SC, CPU, CPU, CPU,
            DC, DC, DC, CL, CL, CL, 0, 0);
}

TEST(Jones, join_matx_scal_scal_singleCPU_doubleCL)
{
    t_join(SCM, SC, SC, CPU, CPU, CPU,
            DCM, DC, DC, CL, CL, CL, 0, 0);
}

TEST(Jones, join_matx_scal_matx_singleCPU_doubleCL)
{
    t_join(SCM, SC, SCM, CPU, CPU, CPU,
            DCM, DC, DCM, CL, CL, CL, 0, 0);
}

TEST(Jones, join_matx_matx_matx_singleCPU_doubleCL)
{
    t_join(SCM, SCM, SCM, CPU, CPU, CPU,
            DCM, DCM, DCM, CL, CL, CL, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_singleCPU_doubleCL)
{
    t_join_in_place(SC, SC, CPU, CPU,
            DC, DC, CL, CL, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_singleCPU_doubleCL)
{
    t_join_in_place(SCM, SC, CPU, CPU,
            DCM, DC, CL, CL, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_singleCPU_doubleCL)
{
    t_join_in_place(SCM, SCM, CPU, CPU,
            DCM, DCM, CL, CL, 0, 0);
}

#endif

#ifdef OSKAR_HAVE_CUDA

// GPU only. //////////////////////////////////////////////////////////////////

TEST(Jones, join_scal_scal_scal_singleGPU_doubleGPU)
{
    t_join(SC, SC, SC, GPU, GPU, GPU,
            DC, DC, DC, GPU, GPU, GPU, 0, 0);
}

TEST(Jones, join_matx_scal_scal_singleGPU_doubleGPU)
{
    t_join(SCM, SC, SC, GPU, GPU, GPU,
            DCM, DC, DC, GPU, GPU, GPU, 0, 0);
}

TEST(Jones, join_matx_scal_matx_singleGPU_doubleGPU)
{
    t_join(SCM, SC, SCM, GPU, GPU, GPU,
            DCM, DC, DCM, GPU, GPU, GPU, 0, 0);
}

TEST(Jones, join_matx_matx_matx_singleGPU_doubleGPU)
{
    t_join(SCM, SCM, SCM, GPU, GPU, GPU,
            DCM, DCM, DCM, GPU, GPU, GPU, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_singleGPU_doubleGPU)
{
    t_join_in_place(SC, SC, GPU, GPU,
            DC, DC, GPU, GPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_singleGPU_doubleGPU)
{
    t_join_in_place(SCM, SC, GPU, GPU,
            DCM, DC, GPU, GPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_singleGPU_doubleGPU)
{
    t_join_in_place(SCM, SCM, GPU, GPU,
            DCM, DCM, GPU, GPU, 0, 0);
}


// Compare CPU and GPU. ///////////////////////////////////////////////////////

// Single GPU, single CPU.
TEST(Jones, join_scal_scal_scal_singleGPU_singleCPU)
{
    t_join(SC, SC, SC, GPU, GPU, GPU,
            SC, SC, SC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_scal_singleGPU_singleCPU)
{
    t_join(SCM, SC, SC, GPU, GPU, GPU,
            SCM, SC, SC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_matx_singleGPU_singleCPU)
{
    t_join(SCM, SC, SCM, GPU, GPU, GPU,
            SCM, SC, SCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_matx_matx_singleGPU_singleCPU)
{
    t_join(SCM, SCM, SCM, GPU, GPU, GPU,
            SCM, SCM, SCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_singleGPU_singleCPU)
{
    t_join_in_place(SC, SC, GPU, GPU,
            SC, SC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_singleGPU_singleCPU)
{
    t_join_in_place(SCM, SC, GPU, GPU,
            SCM, SC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_singleGPU_singleCPU)
{
    t_join_in_place(SCM, SCM, GPU, GPU,
            SCM, SCM, CPU, CPU, 0, 0);
}

// Double GPU, double CPU.
TEST(Jones, join_scal_scal_scal_doubleGPU_doubleCPU)
{
    t_join(DC, DC, DC, GPU, GPU, GPU,
            DC, DC, DC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_scal_doubleGPU_doubleCPU)
{
    t_join(DCM, DC, DC, GPU, GPU, GPU,
            DCM, DC, DC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_matx_doubleGPU_doubleCPU)
{
    t_join(DCM, DC, DCM, GPU, GPU, GPU,
            DCM, DC, DCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_matx_matx_doubleGPU_doubleCPU)
{
    t_join(DCM, DCM, DCM, GPU, GPU, GPU,
            DCM, DCM, DCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_doubleGPU_doubleCPU)
{
    t_join_in_place(DC, DC, GPU, GPU,
            DC, DC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_doubleGPU_doubleCPU)
{
    t_join_in_place(DCM, DC, GPU, GPU,
            DCM, DC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_doubleGPU_doubleCPU)
{
    t_join_in_place(DCM, DCM, GPU, GPU,
            DCM, DCM, CPU, CPU, 0, 0);
}

// Single GPU, double CPU.
TEST(Jones, join_scal_scal_scal_singleGPU_doubleCPU)
{
    t_join(SC, SC, SC, GPU, GPU, GPU,
            DC, DC, DC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_scal_singleGPU_doubleCPU)
{
    t_join(SCM, SC, SC, GPU, GPU, GPU,
            DCM, DC, DC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_matx_singleGPU_doubleCPU)
{
    t_join(SCM, SC, SCM, GPU, GPU, GPU,
            DCM, DC, DCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_matx_matx_singleGPU_doubleCPU)
{
    t_join(SCM, SCM, SCM, GPU, GPU, GPU,
            DCM, DCM, DCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_singleGPU_doubleCPU)
{
    t_join_in_place(SC, SC, GPU, GPU,
            DC, DC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_singleGPU_doubleCPU)
{
    t_join_in_place(SCM, SC, GPU, GPU,
            DCM, DC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_singleGPU_doubleCPU)
{
    t_join_in_place(SCM, SCM, GPU, GPU,
            DCM, DCM, CPU, CPU, 0, 0);
}

// Single CPU, double GPU.
TEST(Jones, join_scal_scal_scal_singleCPU_doubleGPU)
{
    t_join(SC, SC, SC, CPU, CPU, CPU,
            DC, DC, DC, GPU, GPU, GPU, 0, 0);
}

TEST(Jones, join_matx_scal_scal_singleCPU_doubleGPU)
{
    t_join(SCM, SC, SC, CPU, CPU, CPU,
            DCM, DC, DC, GPU, GPU, GPU, 0, 0);
}

TEST(Jones, join_matx_scal_matx_singleCPU_doubleGPU)
{
    t_join(SCM, SC, SCM, CPU, CPU, CPU,
            DCM, DC, DCM, GPU, GPU, GPU, 0, 0);
}

TEST(Jones, join_matx_matx_matx_singleCPU_doubleGPU)
{
    t_join(SCM, SCM, SCM, CPU, CPU, CPU,
            DCM, DCM, DCM, GPU, GPU, GPU, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_singleCPU_doubleGPU)
{
    t_join_in_place(SC, SC, CPU, CPU,
            DC, DC, GPU, GPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_singleCPU_doubleGPU)
{
    t_join_in_place(SCM, SC, CPU, CPU,
            DCM, DC, GPU, GPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_singleCPU_doubleGPU)
{
    t_join_in_place(SCM, SCM, CPU, CPU,
            DCM, DCM, GPU, GPU, 0, 0);
}


// Mixed CPU and GPU. /////////////////////////////////////////////////////////

// Single mixed GPU/CPU, single CPU.
TEST(Jones, join_scal_scal_scal_singleMix_singleCPU)
{
    t_join(SC, SC, SC, GPU, CPU, GPU,
            SC, SC, SC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_scal_singleMix_singleCPU)
{
    t_join(SCM, SC, SC, GPU, CPU, GPU,
            SCM, SC, SC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_matx_singleMix_singleCPU)
{
    t_join(SCM, SC, SCM, GPU, CPU, GPU,
            SCM, SC, SCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_matx_matx_singleMix_singleCPU)
{
    t_join(SCM, SCM, SCM, GPU, CPU, GPU,
            SCM, SCM, SCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_singleMix_singleCPU)
{
    t_join_in_place(SC, SC, GPU, CPU,
            SC, SC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_singleMix_singleCPU)
{
    t_join_in_place(SCM, SC, GPU, CPU,
            SCM, SC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_singleMix_singleCPU)
{
    t_join_in_place(SCM, SCM, GPU, CPU,
            SCM, SCM, CPU, CPU, 0, 0);
}

// Double mixed GPU/CPU, double CPU.
TEST(Jones, join_scal_scal_scal_doubleMix_doubleCPU)
{
    t_join(DC, DC, DC, GPU, CPU, GPU,
            DC, DC, DC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_scal_doubleMix_doubleCPU)
{
    t_join(DCM, DC, DC, GPU, CPU, GPU,
            DCM, DC, DC, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_scal_matx_doubleMix_doubleCPU)
{
    t_join(DCM, DC, DCM, GPU, CPU, GPU,
            DCM, DC, DCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_matx_matx_matx_doubleMix_doubleCPU)
{
    t_join(DCM, DCM, DCM, GPU, CPU, GPU,
            DCM, DCM, DCM, CPU, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_doubleMix_doubleCPU)
{
    t_join_in_place(DC, DC, GPU, CPU,
            DC, DC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_doubleMix_doubleCPU)
{
    t_join_in_place(DCM, DC, GPU, CPU,
            DCM, DC, CPU, CPU, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_doubleMix_doubleCPU)
{
    t_join_in_place(DCM, DCM, GPU, CPU,
            DCM, DCM, CPU, CPU, 0, 0);
}

TEST(Jones, set_ones_singleGPU)
{
    test_ones(OSKAR_SINGLE, OSKAR_GPU);
}

TEST(Jones, set_ones_doubleGPU)
{
    test_ones(OSKAR_DOUBLE, OSKAR_GPU);
}

#endif /* OSKAR_HAVE_CUDA */

TEST(Jones, set_ones_singleCPU)
{
    test_ones(OSKAR_SINGLE, OSKAR_CPU);
}

TEST(Jones, set_ones_doubleCPU)
{
    test_ones(OSKAR_DOUBLE, OSKAR_CPU);
}
