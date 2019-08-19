/*
 * Copyright (c) 2019, The University of Oxford
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

#include "utility/oskar_get_error_string.h"
#include "mem/oskar_mem.h"

TEST(Mem, normalise)
{
    int location = OSKAR_CPU, n = 1000000, status = 0;
    oskar_Mem *cpu, *cpu2, *temp;
#ifdef OSKAR_HAVE_CUDA
    location = OSKAR_GPU;
#endif

    // Create test array and fill with data.
    cpu = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_CPU, n, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    double2* cpu_ = oskar_mem_double2(cpu, &status);
    for (int i = 0; i < n; ++i)
    {
        cpu_[i].x = (double)i;
        cpu_[i].y = 0.0;
    }

    // Normalise on host and device.
    temp = oskar_mem_create_copy(cpu, location, &status);
    oskar_mem_normalise(temp, 0, n, n - 1, &status);
    oskar_mem_normalise(cpu, 0, n, n - 1, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Check for equality.
    cpu2 = oskar_mem_create_copy(temp, OSKAR_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    double2* cpu2_ = oskar_mem_double2(cpu2, &status);
    for (int i = 0; i < n; ++i)
    {
        EXPECT_DOUBLE_EQ(cpu_[i].x, cpu2_[i].x);
        EXPECT_DOUBLE_EQ(cpu_[i].y, cpu2_[i].y);
    }

    // Free memory.
    oskar_mem_free(cpu, &status);
    oskar_mem_free(cpu2, &status);
    oskar_mem_free(temp, &status);
}

