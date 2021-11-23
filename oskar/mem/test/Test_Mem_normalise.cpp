/*
 * Copyright (c) 2019-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "utility/oskar_get_error_string.h"
#include "mem/oskar_mem.h"

TEST(Mem, normalise)
{
    int location = OSKAR_CPU, n = 1000000, status = 0;
    oskar_Mem *cpu = 0, *cpu2 = 0, *temp = 0;
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
