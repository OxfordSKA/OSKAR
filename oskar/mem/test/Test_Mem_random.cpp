/*
 * Copyright (c) 2015-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "mem/oskar_mem.h"
#include "utility/oskar_timer.h"
#include "utility/oskar_get_error_string.h"
#include <cstdio>
#include <cstdlib>

static const bool verbose = false;
static const bool save = false;
static const int n = 544357;

static void report_time(int n, const char* type,
        const char* prec, const char* loc, double sec)
{
    if (verbose)
    {
        printf("Generated %d %s random numbers (%s, %s): %.3f sec\n",
                n, type, prec, loc, sec);
    }
}

TEST(Mem, random_uniform)
{
    int seed = 1;
    int c1 = 437;
    int c2 = 0;
    int c3 = 0xDECAFBAD;
    int status = 0;
    double max_err = 0.0, avg_err = 0.0;
    oskar_Mem* v_cpu_f = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, n, &status);
    oskar_Mem* v_cpu_d = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, n, &status);
#ifdef OSKAR_HAVE_CUDA
    oskar_Mem* v_gpu_f = oskar_mem_create(OSKAR_SINGLE, OSKAR_GPU, n, &status);
    oskar_Mem* v_gpu_d = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU, n, &status);
#endif
#ifdef OSKAR_HAVE_OPENCL
    oskar_Mem* v_cl_f = oskar_mem_create(OSKAR_SINGLE, OSKAR_CL, n, &status);
    oskar_Mem* v_cl_d = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CL, n, &status);
#endif
    oskar_Timer* tmr = oskar_timer_create(OSKAR_TIMER_CUDA);

    // Run on CPU.
    oskar_timer_start(tmr);
    oskar_mem_random_uniform(v_cpu_f, seed, c1, c2, c3, &status);
    report_time(n, "uniform", "single", "CPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_uniform(v_cpu_d, seed, c1, c2, c3, &status);
    report_time(n, "uniform", "double", "CPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

#ifdef OSKAR_HAVE_CUDA
    // Run on GPU with CUDA.
    oskar_timer_start(tmr);
    oskar_mem_random_uniform(v_gpu_f, seed, c1, c2, c3, &status);
    report_time(n, "uniform", "single", "GPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_uniform(v_gpu_f, seed, c1, c2, c3, &status);
    report_time(n, "uniform", "single", "GPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_uniform(v_gpu_d, seed, c1, c2, c3, &status);
    report_time(n, "uniform", "double", "GPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_uniform(v_gpu_d, seed, c1, c2, c3, &status);
    report_time(n, "uniform", "double", "GPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
#endif

#ifdef OSKAR_HAVE_OPENCL
    // Run on OpenCL.
    oskar_timer_start(tmr);
    oskar_mem_random_uniform(v_cl_f, seed, c1, c2, c3, &status);
    report_time(n, "uniform", "single", "OpenCL", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_uniform(v_cl_f, seed, c1, c2, c3, &status);
    report_time(n, "uniform", "single", "OpenCL", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_uniform(v_cl_d, seed, c1, c2, c3, &status);
    report_time(n, "uniform", "double", "OpenCL", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_uniform(v_cl_d, seed, c1, c2, c3, &status);
    report_time(n, "uniform", "double", "OpenCL", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
#endif

    // Check consistency between single and double precision.
    oskar_mem_evaluate_relative_error(v_cpu_f, v_cpu_d, 0,
            &max_err, &avg_err, 0, &status);
    EXPECT_LT(avg_err, 1e-5);

#ifdef OSKAR_HAVE_CUDA
    // Check consistency between CPU and GPU results.
    oskar_mem_evaluate_relative_error(v_gpu_f, v_cpu_f, 0,
            &max_err, &avg_err, 0, &status);
    EXPECT_LT(avg_err, 1e-5);
    oskar_mem_evaluate_relative_error(v_gpu_d, v_cpu_d, 0,
            &max_err, &avg_err, 0, &status);
    EXPECT_LT(avg_err, 1e-10);
#endif

#ifdef OSKAR_HAVE_OPENCL
    // Check consistency between CPU and OpenCL results.
    oskar_mem_evaluate_relative_error(v_cl_f, v_cpu_f, 0,
            &max_err, &avg_err, 0, &status);
    EXPECT_LT(avg_err, 1e-5);
    oskar_mem_evaluate_relative_error(v_cl_d, v_cpu_d, 0,
            &max_err, &avg_err, 0, &status);
    EXPECT_LT(avg_err, 1e-10);
#endif

    if (save)
    {
        size_t num_mem = 2;
        FILE* fhan = fopen("random_uniform.txt", "w");
#ifdef OSKAR_HAVE_CUDA
        num_mem += 2;
#endif
#ifdef OSKAR_HAVE_OPENCL
        num_mem += 2;
#endif
        oskar_mem_save_ascii(fhan, num_mem, 0, n, &status, v_cpu_f, v_cpu_d
#ifdef OSKAR_HAVE_CUDA
                , v_gpu_f, v_gpu_d
#endif
#ifdef OSKAR_HAVE_OPENCL
                , v_cl_f, v_cl_d
#endif
                );
        fclose(fhan);
    }

    // Free memory.
    oskar_mem_free(v_cpu_f, &status);
    oskar_mem_free(v_cpu_d, &status);
#ifdef OSKAR_HAVE_CUDA
    oskar_mem_free(v_gpu_f, &status);
    oskar_mem_free(v_gpu_d, &status);
#endif
#ifdef OSKAR_HAVE_OPENCL
    oskar_mem_free(v_cl_f, &status);
    oskar_mem_free(v_cl_d, &status);
#endif
    oskar_timer_free(tmr);
}

TEST(Mem, random_gaussian)
{
    int seed = 1;
    int c1 = 437;
    int c2 = 0;
    int c3 = 0xDECAFBAD;
    int status = 0;
    double max_err = 0.0, avg_err = 0.0;
    oskar_Mem* v_cpu_f = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, n, &status);
    oskar_Mem* v_cpu_d = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, n, &status);
#ifdef OSKAR_HAVE_CUDA
    oskar_Mem* v_gpu_f = oskar_mem_create(OSKAR_SINGLE, OSKAR_GPU, n, &status);
    oskar_Mem* v_gpu_d = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU, n, &status);
#endif
#ifdef OSKAR_HAVE_OPENCL
    oskar_Mem* v_cl_f = oskar_mem_create(OSKAR_SINGLE, OSKAR_CL, n, &status);
    oskar_Mem* v_cl_d = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CL, n, &status);
#endif
    oskar_Timer* tmr = oskar_timer_create(OSKAR_TIMER_CUDA);

    // Run on CPU.
    oskar_timer_start(tmr);
    oskar_mem_random_gaussian(v_cpu_f, seed, c1, c2, c3, 1.0, &status);
    report_time(n, "Gaussian", "single", "CPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_gaussian(v_cpu_d, seed, c1, c2, c3, 1.0, &status);
    report_time(n, "Gaussian", "double", "CPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

#ifdef OSKAR_HAVE_CUDA
    // Run on GPU with CUDA.
    oskar_timer_start(tmr);
    oskar_mem_random_gaussian(v_gpu_f, seed, c1, c2, c3, 1.0, &status);
    report_time(n, "Gaussian", "single", "GPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_gaussian(v_gpu_f, seed, c1, c2, c3, 1.0, &status);
    report_time(n, "Gaussian", "single", "GPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_gaussian(v_gpu_d, seed, c1, c2, c3, 1.0, &status);
    report_time(n, "Gaussian", "double", "GPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_gaussian(v_gpu_d, seed, c1, c2, c3, 1.0, &status);
    report_time(n, "Gaussian", "double", "GPU", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
#endif

#ifdef OSKAR_HAVE_OPENCL
    // Run on OpenCL.
    oskar_timer_start(tmr);
    oskar_mem_random_gaussian(v_cl_f, seed, c1, c2, c3, 1.0, &status);
    report_time(n, "Gaussian", "single", "OpenCL", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_gaussian(v_cl_f, seed, c1, c2, c3, 1.0, &status);
    report_time(n, "Gaussian", "single", "OpenCL", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_gaussian(v_cl_d, seed, c1, c2, c3, 1.0, &status);
    report_time(n, "Gaussian", "double", "OpenCL", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_timer_start(tmr);
    oskar_mem_random_gaussian(v_cl_d, seed, c1, c2, c3, 1.0, &status);
    report_time(n, "Gaussian", "double", "OpenCL", oskar_timer_elapsed(tmr));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
#endif

    // Check consistency between single and double precision.
    oskar_mem_evaluate_relative_error(v_cpu_f, v_cpu_d, 0,
            &max_err, &avg_err, 0, &status);
    EXPECT_LT(avg_err, 1e-5);

#ifdef OSKAR_HAVE_CUDA
    // Check consistency between CPU and GPU results.
    oskar_mem_evaluate_relative_error(v_gpu_f, v_cpu_f, 0,
            &max_err, &avg_err, 0, &status);
    EXPECT_LT(avg_err, 1e-5);
    oskar_mem_evaluate_relative_error(v_gpu_d, v_cpu_d, 0,
            &max_err, &avg_err, 0, &status);
    EXPECT_LT(avg_err, 1e-10);
#endif

#ifdef OSKAR_HAVE_OPENCL
    // Check consistency between CPU and OpenCL results.
    oskar_mem_evaluate_relative_error(v_cl_f, v_cpu_f, 0,
            &max_err, &avg_err, 0, &status);
    EXPECT_LT(avg_err, 1e-5);
    oskar_mem_evaluate_relative_error(v_cl_d, v_cpu_d, 0,
            &max_err, &avg_err, 0, &status);
    EXPECT_LT(avg_err, 1e-10);
#endif

    if (save)
    {
        size_t num_mem = 2;
        FILE* fhan = fopen("random_gaussian.txt", "w");
#ifdef OSKAR_HAVE_CUDA
        num_mem += 2;
#endif
#ifdef OSKAR_HAVE_OPENCL
        num_mem += 2;
#endif
        oskar_mem_save_ascii(fhan, num_mem, 0, n, &status, v_cpu_f, v_cpu_d
#ifdef OSKAR_HAVE_CUDA
                , v_gpu_f, v_gpu_d
#endif
#ifdef OSKAR_HAVE_OPENCL
                , v_cl_f, v_cl_d
#endif
                );
        fclose(fhan);
    }

    // Free memory.
    oskar_mem_free(v_cpu_f, &status);
    oskar_mem_free(v_cpu_d, &status);
#ifdef OSKAR_HAVE_CUDA
    oskar_mem_free(v_gpu_f, &status);
    oskar_mem_free(v_gpu_d, &status);
#endif
#ifdef OSKAR_HAVE_OPENCL
    oskar_mem_free(v_cl_f, &status);
    oskar_mem_free(v_cl_d, &status);
#endif
    oskar_timer_free(tmr);
}

TEST(Mem, random_gaussian_accum)
{
    int status = 0;
    int seed = 1;
    int blocksize = 256;
    int rounds = 128;
#ifdef OSKAR_HAVE_CUDA
    int location = OSKAR_GPU;
#else
    int location = OSKAR_CPU;
#endif
    oskar_Mem* block = oskar_mem_create(OSKAR_SINGLE, location,
            blocksize, &status);
    oskar_Mem* total = oskar_mem_create(OSKAR_SINGLE, location,
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
        oskar_mem_save_ascii(fhan, 1, 0, blocksize * rounds, &status, total);
        fclose(fhan);
    }

    oskar_mem_free(block, &status);
    oskar_mem_free(total, &status);
}
