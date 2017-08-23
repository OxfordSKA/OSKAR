/*
 * Copyright (c) 2017, The University of Oxford
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

#include <cstdlib>

#ifdef OSKAR_HAVE_CUDA
int dev_loc = OSKAR_GPU;
#else
int dev_loc = OSKAR_CPU;
#endif

TEST(prefix_sum, test)
{
    int n = 100000, status = 0, exclusive = 1;
    oskar_Mem* in_cpu = oskar_mem_create(OSKAR_INT, OSKAR_CPU, n, &status);
    oskar_Mem* out_cpu = oskar_mem_create(OSKAR_INT, OSKAR_CPU, n, &status);
    oskar_Mem* out_dev = oskar_mem_create(OSKAR_INT, dev_loc, n, &status);
    oskar_Timer* tmr = oskar_timer_create(OSKAR_TIMER_NATIVE);

    // Fill input with random integers from 0 to 9.
    int* t = oskar_mem_int(in_cpu, &status);
    srand(1556);
    for (int i = 0; i < n; ++i)
        t[i] = (int) (10.0 * rand() / ((double) RAND_MAX));
    t[0] = 3;
    oskar_Mem* in_dev = oskar_mem_create_copy(in_cpu, dev_loc, &status);

    // Prefix sum using CPU.
    oskar_timer_start(tmr);
    oskar_prefix_sum(n, in_cpu, out_cpu, 0, exclusive, &status);
//    printf("Prefix sum on CPU took %.3f sec\n", oskar_timer_elapsed(tmr));

    // Prefix sum using device.
    oskar_timer_start(tmr);
    oskar_prefix_sum(n, in_dev, out_dev, 0, exclusive, &status);
//    printf("Prefix sum on device took %.3f sec\n", oskar_timer_elapsed(tmr));

    // Check results.
    oskar_Mem* out_cmp = oskar_mem_create_copy(out_dev, OSKAR_CPU, &status);
    EXPECT_EQ(0, oskar_mem_different(out_cpu, out_cmp, n, &status));

    // Clean up.
    oskar_timer_free(tmr);
    oskar_mem_free(in_cpu, &status);
    oskar_mem_free(in_dev, &status);
    oskar_mem_free(out_cpu, &status);
    oskar_mem_free(out_dev, &status);
    oskar_mem_free(out_cmp, &status);
}
