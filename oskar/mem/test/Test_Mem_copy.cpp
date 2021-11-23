/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "utility/oskar_get_error_string.h"
#include "mem/oskar_mem.h"

TEST(Mem, copy_to_device)
{
    int location = OSKAR_CPU, n = 100, status = 0;
    oskar_Mem *cpu = 0, *cpu2 = 0, *temp = 0;
#ifdef OSKAR_HAVE_CUDA
    location = OSKAR_GPU;
#endif

    // Create test array and fill with data.
    cpu = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, n, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    double* cpu_ = oskar_mem_double(cpu, &status);
    for (int i = 0; i < n; ++i)
    {
        cpu_[i] = (double)i;
    }

    // Copy to device.
    temp = oskar_mem_create_copy(cpu, location, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    // Copy back and check for equality.
    cpu2 = oskar_mem_create_copy(temp, OSKAR_CPU, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    double* cpu2_ = oskar_mem_double(cpu2, &status);
    for (int i = 0; i < n; ++i)
    {
        EXPECT_DOUBLE_EQ(cpu_[i], cpu2_[i]);
    }

    // Free memory.
    oskar_mem_free(cpu, &status);
    oskar_mem_free(cpu2, &status);
    oskar_mem_free(temp, &status);
}
