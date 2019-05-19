/*
 * Copyright (c) 2017-2019, The University of Oxford
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

#include "math/oskar_prefix_sum.h"
#include "utility/oskar_timer.h"
#include "utility/oskar_device.h"

#include <cstdlib>

static const bool save = true;

void run_test(const oskar_Mem* in_cpu, const char* fname)
{
    int status = 0;
    const int n = (int) oskar_mem_length(in_cpu);

    // Run on CPU.
    oskar_Mem* out_cpu = oskar_mem_create(OSKAR_INT, OSKAR_CPU, n + 1, &status);
    oskar_Timer* tmr = oskar_timer_create(OSKAR_TIMER_NATIVE);
    oskar_timer_start(tmr);
    oskar_prefix_sum(n, in_cpu, out_cpu, &status);
    EXPECT_EQ(0, status);
    printf("Prefix sum on CPU took %.3f sec\n", oskar_timer_elapsed(tmr));
    oskar_timer_free(tmr);

#ifdef OSKAR_HAVE_CUDA
    // Run on GPU with CUDA.
    oskar_Mem* in_gpu = oskar_mem_create_copy(in_cpu, OSKAR_GPU, &status);
    oskar_Mem* out_gpu = oskar_mem_create(OSKAR_INT, OSKAR_GPU, n + 1, &status);
    tmr = oskar_timer_create(OSKAR_TIMER_CUDA);
    oskar_timer_start(tmr);
    oskar_prefix_sum(n, in_gpu, out_gpu, &status);
    EXPECT_EQ(0, status);
    printf("Prefix sum on GPU took %.3f sec\n", oskar_timer_elapsed(tmr));
    oskar_timer_free(tmr);

    // Check consistency between CPU and GPU results.
    oskar_Mem* out_cmp_gpu = oskar_mem_create_copy(out_gpu, OSKAR_CPU, &status);
    EXPECT_EQ(0, oskar_mem_different(out_cpu, out_cmp_gpu, n + 1, &status));
#endif

#ifdef OSKAR_HAVE_OPENCL
    // Run on OpenCL.
    oskar_Mem* in_cl = oskar_mem_create_copy(in_cpu, OSKAR_CL, &status);
    oskar_Mem* out_cl = oskar_mem_create(OSKAR_INT, OSKAR_CL, n + 1, &status);
    tmr = oskar_timer_create(OSKAR_TIMER_CL);
    char* device_name = oskar_device_name(OSKAR_CL, 0);
    oskar_timer_start(tmr);
    oskar_prefix_sum(n, in_cl, out_cl, &status);
    EXPECT_EQ(0, status);
    printf("Prefix sum on OpenCL device '%s' took %.3f sec\n", device_name,
            oskar_timer_elapsed(tmr));
    free(device_name);
    oskar_timer_free(tmr);

    // Check consistency between CPU and OpenCL results.
    oskar_Mem* out_cmp_cl = oskar_mem_create_copy(out_cl, OSKAR_CPU, &status);
    EXPECT_EQ(0, oskar_mem_different(out_cpu, out_cmp_cl, n + 1, &status));
#endif

    if (save)
    {
        size_t num_mem = 1;
        FILE* fhan = fopen(fname, "w");
#ifdef OSKAR_HAVE_CUDA
        num_mem += 1;
#endif
#ifdef OSKAR_HAVE_OPENCL
        num_mem += 1;
#endif
        oskar_mem_save_ascii(fhan, num_mem, 0, n + 1, &status, out_cpu
#ifdef OSKAR_HAVE_CUDA
                , out_cmp_gpu
#endif
#ifdef OSKAR_HAVE_OPENCL
                , out_cmp_cl
#endif
                );
        fclose(fhan);
    }

    // Clean up.
#ifdef OSKAR_HAVE_CUDA
    oskar_mem_free(in_gpu, &status);
    oskar_mem_free(out_gpu, &status);
    oskar_mem_free(out_cmp_gpu, &status);
#endif
#ifdef OSKAR_HAVE_OPENCL
    oskar_mem_free(in_cl, &status);
    oskar_mem_free(out_cl, &status);
    oskar_mem_free(out_cmp_cl, &status);
#endif
    oskar_mem_free(out_cpu, &status);
}

TEST(prefix_sum, test1)
{
    int n = 100000, status = 0;
    oskar_Mem* in = oskar_mem_create(OSKAR_INT, OSKAR_CPU, n, &status);
    int* t = oskar_mem_int(in, &status);
    srand(1556);
    for (int i = 0; i < n; ++i)
        t[i] = (int) (10.0 * rand() / ((double) RAND_MAX));
    t[0] = 3;
    run_test(in, "prefix_sum_test1.txt");
    oskar_mem_free(in, &status);
}

TEST(prefix_sum, test2)
{
    int n = 4, status = 0;
    oskar_Mem* in = oskar_mem_create(OSKAR_INT, OSKAR_CPU, n, &status);
    int* t = oskar_mem_int(in, &status);
    for (int i = 0; i < n; ++i) t[i] = i + 1;
    run_test(in, "prefix_sum_test2.txt");
    oskar_mem_free(in, &status);
}
