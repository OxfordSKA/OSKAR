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

#include <oskar_mem.h>
#include <oskar_timer.h>
#include <oskar_get_error_string.h>
#include <cstdio>
#include <cstdlib>

static const bool verbose = false;
static const bool save = false;

static void report_time(int n, const char* type,
        const char* prec, const char* loc, double sec)
{
    if (verbose)
        printf("Generated %d %s random numbers (%s, %s): %.3f sec\n",
                n, type, prec, loc, sec);
}

TEST(Mem, random_uniform)
{
    int seed = 1;
    int c1 = 437;
    int c2 = 0;
    int c3 = 0xDECAFBAD;
    int n = 544357;
    int status = 0;
    double max_err = 0.0, avg_err = 0.0;
    oskar_Mem* v_cpu_f = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, n, &status);
    oskar_Mem* v_gpu_f = oskar_mem_create(OSKAR_SINGLE, OSKAR_GPU, n, &status);
    oskar_Mem* v_cpu_d = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, n, &status);
    oskar_Mem* v_gpu_d = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU, n, &status);
    oskar_Timer* tmr = oskar_timer_create(OSKAR_TIMER_CUDA);

    // Run in single precision.
    oskar_timer_start(tmr);
    oskar_mem_random_uniform(v_cpu_f, seed, c1, c2, c3, &status);
    report_time(n, "uniform", "single", "CPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_uniform(v_gpu_f, seed, c1, c2, c3, &status);
    report_time(n, "uniform", "single", "GPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check consistency between CPU and GPU results.
    oskar_mem_evaluate_relative_error(v_gpu_f, v_cpu_f, 0,
            &max_err, &avg_err, 0, &status);
    EXPECT_LT(max_err, 1e-5);
    EXPECT_LT(avg_err, 1e-5);

    // Run in double precision.
    oskar_timer_start(tmr);
    oskar_mem_random_uniform(v_cpu_d, seed, c1, c2, c3, &status);
    report_time(n, "uniform", "double", "CPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_uniform(v_gpu_d, seed, c1, c2, c3, &status);
    report_time(n, "uniform", "double", "GPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check consistency between CPU and GPU results.
    oskar_mem_evaluate_relative_error(v_gpu_d, v_cpu_d, 0,
            &max_err, &avg_err, 0, &status);
    EXPECT_LT(max_err, 1e-10);
    EXPECT_LT(avg_err, 1e-10);

    // Check consistency between single and double precision.
    oskar_mem_evaluate_relative_error(v_cpu_f, v_cpu_d, 0,
            &max_err, &avg_err, 0, &status);
    EXPECT_LT(max_err, 1e-5);
    EXPECT_LT(avg_err, 1e-5);

    if (save)
    {
        FILE* fhan = fopen("random_uniform.txt", "w");
        oskar_mem_save_ascii(fhan, 4, n, &status,
                v_cpu_f, v_gpu_f, v_cpu_d, v_gpu_d);
        fclose(fhan);
    }

    // Free memory.
    oskar_mem_free(v_cpu_f, &status);
    oskar_mem_free(v_gpu_f, &status);
    oskar_mem_free(v_cpu_d, &status);
    oskar_mem_free(v_gpu_d, &status);
    oskar_timer_free(tmr);
}

TEST(Mem, random_gaussian)
{
    int seed = 1;
    int c1 = 437;
    int c2 = 0;
    int c3 = 0xDECAFBAD;
    int n = 267587;
    int status = 0;
    double max_err = 0.0, avg_err = 0.0;
    oskar_Mem* v_cpu_f = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, n, &status);
    oskar_Mem* v_gpu_f = oskar_mem_create(OSKAR_SINGLE, OSKAR_GPU, n, &status);
    oskar_Mem* v_cpu_d = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, n, &status);
    oskar_Mem* v_gpu_d = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU, n, &status);
    oskar_Timer* tmr = oskar_timer_create(OSKAR_TIMER_CUDA);

    // Run in single precision.
    oskar_timer_start(tmr);
    oskar_mem_random_gaussian(v_cpu_f, seed, c1, c2, c3, 1.0, &status);
    report_time(n, "Gaussian", "single", "CPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_gaussian(v_gpu_f, seed, c1, c2, c3, 1.0, &status);
    report_time(n, "Gaussian", "single", "GPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check consistency between CPU and GPU results.
    oskar_mem_evaluate_relative_error(v_gpu_f, v_cpu_f, 0,
            &max_err, &avg_err, 0, &status);
    EXPECT_LT(avg_err, 1e-5);

    // Run in double precision.
    oskar_timer_start(tmr);
    oskar_mem_random_gaussian(v_cpu_d, seed, c1, c2, c3, 1.0, &status);
    report_time(n, "Gaussian", "double", "CPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_gaussian(v_gpu_d, seed, c1, c2, c3, 1.0, &status);
    report_time(n, "Gaussian", "double", "GPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check consistency between CPU and GPU results.
    oskar_mem_evaluate_relative_error(v_gpu_d, v_cpu_d, 0,
            &max_err, &avg_err, 0, &status);
    EXPECT_LT(avg_err, 1e-10);

    // Check consistency between single and double precision.
    oskar_mem_evaluate_relative_error(v_cpu_f, v_cpu_d, 0,
            &max_err, &avg_err, 0, &status);
    EXPECT_LT(avg_err, 1e-5);

    if (save)
    {
        FILE* fhan = fopen("random_gaussian.txt", "w");
        oskar_mem_save_ascii(fhan, 4, n, &status,
                v_cpu_f, v_gpu_f, v_cpu_d, v_gpu_d);
        fclose(fhan);
    }

    // Free memory.
    oskar_mem_free(v_cpu_f, &status);
    oskar_mem_free(v_gpu_f, &status);
    oskar_mem_free(v_cpu_d, &status);
    oskar_mem_free(v_gpu_d, &status);
    oskar_timer_free(tmr);
}

TEST(Mem, random_gaussian_accum)
{
    int status = 0;
    int seed = 1;
    int blocksize = 256;
    int rounds = 10240;
    oskar_Mem* block = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU,
            blocksize, &status);
    oskar_Mem* total = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU,
            blocksize * rounds, &status);

    for (int i = 0; i < rounds; ++i)
    {
        oskar_mem_random_gaussian(block, seed, i, 0, 0, 1.0, &status);
        oskar_mem_copy_contents(total, block,
                i * blocksize, 0, blocksize, &status);
    }
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    if (save)
    {
        FILE* fhan = fopen("random_gaussian_accum.txt", "w");
        oskar_mem_save_ascii(fhan, 1, blocksize * rounds, &status, total);
        fclose(fhan);
    }

    oskar_mem_free(block, &status);
    oskar_mem_free(total, &status);
}
