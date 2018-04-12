/*
 * Copyright (c) 2017-2018, The University of Oxford
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
#include "utility/oskar_cl_utils.h"

#include <cstdlib>

static const bool save = true;

TEST(prefix_sum, test)
{
    int n = 100000, status = 0, exclusive = 1;
    oskar_Mem* in_cpu = oskar_mem_create(OSKAR_INT, OSKAR_CPU, n, &status);
    oskar_Mem* out_cpu = oskar_mem_create(OSKAR_INT, OSKAR_CPU, n, &status);
    oskar_Timer* tmr = oskar_timer_create(OSKAR_TIMER_NATIVE);

    // Fill input with random integers from 0 to 9.
    int* t = oskar_mem_int(in_cpu, &status);
    srand(1556);
    for (int i = 0; i < n; ++i)
        t[i] = (int) (10.0 * rand() / ((double) RAND_MAX));
    t[0] = 3;

    // Run on CPU.
    oskar_timer_start(tmr);
    oskar_prefix_sum(n, in_cpu, out_cpu, 0, exclusive, &status);
    EXPECT_EQ(0, status);
    printf("Prefix sum on CPU took %.3f sec\n", oskar_timer_elapsed(tmr));

#ifdef OSKAR_HAVE_CUDA
    // Run on GPU with CUDA.
    oskar_Mem* in_gpu = oskar_mem_create_copy(in_cpu, OSKAR_GPU, &status);
    oskar_Mem* out_gpu = oskar_mem_create(OSKAR_INT, OSKAR_GPU, n, &status);
    oskar_timer_start(tmr);
    oskar_prefix_sum(n, in_gpu, out_gpu, 0, exclusive, &status);
    EXPECT_EQ(0, status);
    printf("Prefix sum on GPU took %.3f sec\n", oskar_timer_elapsed(tmr));

    // Check consistency between CPU and GPU results.
    oskar_Mem* out_cmp_gpu = oskar_mem_create_copy(out_gpu, OSKAR_CPU, &status);
    EXPECT_EQ(0, oskar_mem_different(out_cpu, out_cmp_gpu, n, &status));
#endif

#ifdef OSKAR_HAVE_OPENCL
    // Run on OpenCL.
    oskar_Mem* in_cl = oskar_mem_create_copy(in_cpu, OSKAR_CL, &status);
    oskar_Mem* out_cl = oskar_mem_create(OSKAR_INT, OSKAR_CL, n, &status);
    oskar_timer_start(tmr);
    printf("Using %s\n", oskar_cl_device_name());
    oskar_prefix_sum(n, in_cl, out_cl, 0, exclusive, &status);
    EXPECT_EQ(0, status);
    printf("Prefix sum on OpenCL took %.3f sec\n", oskar_timer_elapsed(tmr));

    // Check consistency between CPU and OpenCL results.
    oskar_Mem* out_cmp_cl = oskar_mem_create_copy(out_cl, OSKAR_CPU, &status);
    EXPECT_EQ(0, oskar_mem_different(out_cpu, out_cmp_cl, n, &status));
#endif

    if (save)
    {
        size_t num_mem = 1;
        FILE* fhan = fopen("prefix_sum_test.txt", "w");
#ifdef OSKAR_HAVE_CUDA
        num_mem += 1;
#endif
#ifdef OSKAR_HAVE_OPENCL
        num_mem += 1;
#endif
        oskar_mem_save_ascii(fhan, num_mem, n, &status, out_cpu
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
    oskar_timer_free(tmr);
    oskar_mem_free(in_cpu, &status);
    oskar_mem_free(out_cpu, &status);
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
}
