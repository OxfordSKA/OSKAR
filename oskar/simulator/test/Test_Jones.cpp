/*
 * Copyright (c) 2011-2015, The University of Oxford
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

#include <oskar_jones.h>
#include <oskar_get_error_string.h>
#include <oskar_mem.h>
#include <oskar_vector_types.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>

#define LC   OSKAR_CPU
#define LG   OSKAR_GPU
#define TSC  OSKAR_SINGLE_COMPLEX
#define TSCM OSKAR_SINGLE_COMPLEX_MATRIX
#define TDC  OSKAR_DOUBLE_COMPLEX
#define TDCM OSKAR_DOUBLE_COMPLEX_MATRIX

static void check_values(const oskar_Mem* approx, const oskar_Mem* accurate)
{
    int status = 0;
    double min_rel_error, max_rel_error, avg_rel_error, std_rel_error, tol;
    oskar_mem_evaluate_relative_error(approx, accurate, &min_rel_error,
            &max_rel_error, &avg_rel_error, &std_rel_error, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    tol = oskar_mem_is_double(approx) &&
            oskar_mem_is_double(accurate) ? 1e-12 : 1e-5;
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

static const int sources = 1000;
static const int stations = 30;

static void t_join(int out_typeA, int in_type1A, int in_type2A,
        int out_locA, int in_loc1A, int in_loc2A,
        int out_typeB, int in_type1B, int in_type2B,
        int out_locB, int in_loc1B, int in_loc2B,
        int expectedA, int expectedB)
{
    int status = 0;
    oskar_Timer *timerA, *timerB;
//    double timeA, timeB;
    oskar_Jones *in1, *in2, *outA, *outB;

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
    oskar_Timer *timerA, *timerB;
//    double timeA, timeB;
    oskar_Jones *in1A, *in2A, *in1B, *in2B;

    // Create the timers.
    timerA = oskar_timer_create(in_type1A == OSKAR_GPU ?
            OSKAR_TIMER_CUDA : OSKAR_TIMER_NATIVE);
    timerB = oskar_timer_create(in_type1B == OSKAR_GPU ?
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


// CPU only. //////////////////////////////////////////////////////////////////

TEST(Jones, join_scal_scal_scal_singleCPU_doubleCPU)
{
    t_join(TSC, TSC, TSC, LC, LC, LC,
            TDC, TDC, TDC, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_scal_scal_singleCPU_doubleCPU)
{
    t_join(TSCM, TSC, TSC, LC, LC, LC,
            TDCM, TDC, TDC, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_scal_matx_singleCPU_doubleCPU)
{
    t_join(TSCM, TSC, TSCM, LC, LC, LC,
            TDCM, TDC, TDCM, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_matx_matx_singleCPU_doubleCPU)
{
    t_join(TSCM, TSCM, TSCM, LC, LC, LC,
            TDCM, TDCM, TDCM, LC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_singleCPU_doubleCPU)
{
    t_join_in_place(TSC, TSC, LC, LC,
            TDC, TDC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_singleCPU_doubleCPU)
{
    t_join_in_place(TSCM, TSC, LC, LC,
            TDCM, TDC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_singleCPU_doubleCPU)
{
    t_join_in_place(TSCM, TSCM, LC, LC,
            TDCM, TDCM, LC, LC, 0, 0);
}

#ifdef OSKAR_HAVE_CUDA


// GPU only. //////////////////////////////////////////////////////////////////

TEST(Jones, join_scal_scal_scal_singleGPU_doubleGPU)
{
    t_join(TSC, TSC, TSC, LG, LG, LG,
            TDC, TDC, TDC, LG, LG, LG, 0, 0);
}

TEST(Jones, join_matx_scal_scal_singleGPU_doubleGPU)
{
    t_join(TSCM, TSC, TSC, LG, LG, LG,
            TDCM, TDC, TDC, LG, LG, LG, 0, 0);
}

TEST(Jones, join_matx_scal_matx_singleGPU_doubleGPU)
{
    t_join(TSCM, TSC, TSCM, LG, LG, LG,
            TDCM, TDC, TDCM, LG, LG, LG, 0, 0);
}

TEST(Jones, join_matx_matx_matx_singleGPU_doubleGPU)
{
    t_join(TSCM, TSCM, TSCM, LG, LG, LG,
            TDCM, TDCM, TDCM, LG, LG, LG, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_singleGPU_doubleGPU)
{
    t_join_in_place(TSC, TSC, LG, LG,
            TDC, TDC, LG, LG, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_singleGPU_doubleGPU)
{
    t_join_in_place(TSCM, TSC, LG, LG,
            TDCM, TDC, LG, LG, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_singleGPU_doubleGPU)
{
    t_join_in_place(TSCM, TSCM, LG, LG,
            TDCM, TDCM, LG, LG, 0, 0);
}


// Compare CPU and GPU. ///////////////////////////////////////////////////////

// Single GPU, single CPU.
TEST(Jones, join_scal_scal_scal_singleGPU_singleCPU)
{
    t_join(TSC, TSC, TSC, LG, LG, LG,
            TSC, TSC, TSC, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_scal_scal_singleGPU_singleCPU)
{
    t_join(TSCM, TSC, TSC, LG, LG, LG,
            TSCM, TSC, TSC, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_scal_matx_singleGPU_singleCPU)
{
    t_join(TSCM, TSC, TSCM, LG, LG, LG,
            TSCM, TSC, TSCM, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_matx_matx_singleGPU_singleCPU)
{
    t_join(TSCM, TSCM, TSCM, LG, LG, LG,
            TSCM, TSCM, TSCM, LC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_singleGPU_singleCPU)
{
    t_join_in_place(TSC, TSC, LG, LG,
            TSC, TSC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_singleGPU_singleCPU)
{
    t_join_in_place(TSCM, TSC, LG, LG,
            TSCM, TSC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_singleGPU_singleCPU)
{
    t_join_in_place(TSCM, TSCM, LG, LG,
            TSCM, TSCM, LC, LC, 0, 0);
}

// Double GPU, double CPU.
TEST(Jones, join_scal_scal_scal_doubleGPU_doubleCPU)
{
    t_join(TDC, TDC, TDC, LG, LG, LG,
            TDC, TDC, TDC, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_scal_scal_doubleGPU_doubleCPU)
{
    t_join(TDCM, TDC, TDC, LG, LG, LG,
            TDCM, TDC, TDC, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_scal_matx_doubleGPU_doubleCPU)
{
    t_join(TDCM, TDC, TDCM, LG, LG, LG,
            TDCM, TDC, TDCM, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_matx_matx_doubleGPU_doubleCPU)
{
    t_join(TDCM, TDCM, TDCM, LG, LG, LG,
            TDCM, TDCM, TDCM, LC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_doubleGPU_doubleCPU)
{
    t_join_in_place(TDC, TDC, LG, LG,
            TDC, TDC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_doubleGPU_doubleCPU)
{
    t_join_in_place(TDCM, TDC, LG, LG,
            TDCM, TDC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_doubleGPU_doubleCPU)
{
    t_join_in_place(TDCM, TDCM, LG, LG,
            TDCM, TDCM, LC, LC, 0, 0);
}

// Single GPU, double CPU.
TEST(Jones, join_scal_scal_scal_singleGPU_doubleCPU)
{
    t_join(TSC, TSC, TSC, LG, LG, LG,
            TDC, TDC, TDC, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_scal_scal_singleGPU_doubleCPU)
{
    t_join(TSCM, TSC, TSC, LG, LG, LG,
            TDCM, TDC, TDC, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_scal_matx_singleGPU_doubleCPU)
{
    t_join(TSCM, TSC, TSCM, LG, LG, LG,
            TDCM, TDC, TDCM, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_matx_matx_singleGPU_doubleCPU)
{
    t_join(TSCM, TSCM, TSCM, LG, LG, LG,
            TDCM, TDCM, TDCM, LC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_singleGPU_doubleCPU)
{
    t_join_in_place(TSC, TSC, LG, LG,
            TDC, TDC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_singleGPU_doubleCPU)
{
    t_join_in_place(TSCM, TSC, LG, LG,
            TDCM, TDC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_singleGPU_doubleCPU)
{
    t_join_in_place(TSCM, TSCM, LG, LG,
            TDCM, TDCM, LC, LC, 0, 0);
}

// Single CPU, double GPU.
TEST(Jones, join_scal_scal_scal_singleCPU_doubleGPU)
{
    t_join(TSC, TSC, TSC, LC, LC, LC,
            TDC, TDC, TDC, LG, LG, LG, 0, 0);
}

TEST(Jones, join_matx_scal_scal_singleCPU_doubleGPU)
{
    t_join(TSCM, TSC, TSC, LC, LC, LC,
            TDCM, TDC, TDC, LG, LG, LG, 0, 0);
}

TEST(Jones, join_matx_scal_matx_singleCPU_doubleGPU)
{
    t_join(TSCM, TSC, TSCM, LC, LC, LC,
            TDCM, TDC, TDCM, LG, LG, LG, 0, 0);
}

TEST(Jones, join_matx_matx_matx_singleCPU_doubleGPU)
{
    t_join(TSCM, TSCM, TSCM, LC, LC, LC,
            TDCM, TDCM, TDCM, LG, LG, LG, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_singleCPU_doubleGPU)
{
    t_join_in_place(TSC, TSC, LC, LC,
            TDC, TDC, LG, LG, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_singleCPU_doubleGPU)
{
    t_join_in_place(TSCM, TSC, LC, LC,
            TDCM, TDC, LG, LG, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_singleCPU_doubleGPU)
{
    t_join_in_place(TSCM, TSCM, LC, LC,
            TDCM, TDCM, LG, LG, 0, 0);
}


// Mixed CPU and GPU. /////////////////////////////////////////////////////////

// Single mixed GPU/CPU, single CPU.
TEST(Jones, join_scal_scal_scal_singleMix_singleCPU)
{
    t_join(TSC, TSC, TSC, LG, LC, LG,
            TSC, TSC, TSC, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_scal_scal_singleMix_singleCPU)
{
    t_join(TSCM, TSC, TSC, LG, LC, LG,
            TSCM, TSC, TSC, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_scal_matx_singleMix_singleCPU)
{
    t_join(TSCM, TSC, TSCM, LG, LC, LG,
            TSCM, TSC, TSCM, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_matx_matx_singleMix_singleCPU)
{
    t_join(TSCM, TSCM, TSCM, LG, LC, LG,
            TSCM, TSCM, TSCM, LC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_singleMix_singleCPU)
{
    t_join_in_place(TSC, TSC, LG, LC,
            TSC, TSC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_singleMix_singleCPU)
{
    t_join_in_place(TSCM, TSC, LG, LC,
            TSCM, TSC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_singleMix_singleCPU)
{
    t_join_in_place(TSCM, TSCM, LG, LC,
            TSCM, TSCM, LC, LC, 0, 0);
}

// Double mixed GPU/CPU, double CPU.
TEST(Jones, join_scal_scal_scal_doubleMix_doubleCPU)
{
    t_join(TDC, TDC, TDC, LG, LC, LG,
            TDC, TDC, TDC, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_scal_scal_doubleMix_doubleCPU)
{
    t_join(TDCM, TDC, TDC, LG, LC, LG,
            TDCM, TDC, TDC, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_scal_matx_doubleMix_doubleCPU)
{
    t_join(TDCM, TDC, TDCM, LG, LC, LG,
            TDCM, TDC, TDCM, LC, LC, LC, 0, 0);
}

TEST(Jones, join_matx_matx_matx_doubleMix_doubleCPU)
{
    t_join(TDCM, TDCM, TDCM, LG, LC, LG,
            TDCM, TDCM, TDCM, LC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_scal_scal_scal_doubleMix_doubleCPU)
{
    t_join_in_place(TDC, TDC, LG, LC,
            TDC, TDC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_scal_doubleMix_doubleCPU)
{
    t_join_in_place(TDCM, TDC, LG, LC,
            TDCM, TDC, LC, LC, 0, 0);
}

TEST(Jones, join_in_place_matx_matx_matx_doubleMix_doubleCPU)
{
    t_join_in_place(TDCM, TDCM, LG, LC,
            TDCM, TDCM, LC, LC, 0, 0);
}

static void test_ones(int precision, int location)
{
    oskar_Jones *jones, *temp = 0, *j_ptr;
    int status = 0, num_stations, num_sources;

    // Test scalar complex type.
    jones = oskar_jones_create(precision | OSKAR_COMPLEX, location,
            stations, sources, &status);
    ASSERT_EQ(0, status);
    oskar_jones_set_real_scalar(jones, 1.0, &status);
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
        float2* p = oskar_jones_float2(j_ptr, &status);
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
        double2* p = oskar_jones_double2(j_ptr, &status);
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
        oskar_jones_free(temp, &status);
    oskar_jones_free(jones, &status);
    ASSERT_EQ(0, status);

    // Test matrix complex type.
    jones = oskar_jones_create(precision | OSKAR_COMPLEX | OSKAR_MATRIX,
            location, stations, sources, &status);
    ASSERT_EQ(0, status);
    oskar_jones_set_real_scalar(jones, 1.0, &status);
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
        float4c* p = oskar_jones_float4c(j_ptr, &status);
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
        double4c* p = oskar_jones_double4c(j_ptr, &status);
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
        oskar_jones_free(temp, &status);
    oskar_jones_free(jones, &status);
    ASSERT_EQ(0, status);
}

TEST(Jones, set_ones_singleCPU)
{
    test_ones(OSKAR_SINGLE, OSKAR_CPU);
}

TEST(Jones, set_ones_singleGPU)
{
    test_ones(OSKAR_SINGLE, OSKAR_GPU);
}

TEST(Jones, set_ones_doubleCPU)
{
    test_ones(OSKAR_DOUBLE, OSKAR_CPU);
}

TEST(Jones, set_ones_doubleGPU)
{
    test_ones(OSKAR_DOUBLE, OSKAR_GPU);
}

#endif
