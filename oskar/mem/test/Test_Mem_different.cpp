/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "utility/oskar_get_error_string.h"
#include "mem/oskar_mem.h"


TEST(Mem, different_none)
{
    // Test two memory blocks that are the same.
    int status = 0;
    oskar_Mem *one = 0, *two = 0;
    one = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 20, &status);
    two = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 20, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_set_value_real(one, 4.4, 0, 20, &status);
    oskar_mem_set_value_real(two, 4.4, 0, 20, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ((int)OSKAR_FALSE, oskar_mem_different(one, two, 0, &status));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_free(one, &status);
    oskar_mem_free(two, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}


TEST(Mem, different_all)
{
    // Test two memory blocks that are different.
    int status = 0;
    oskar_Mem *one = 0, *two = 0;
    one = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 20, &status);
    two = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 20, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_set_value_real(one, 4.4, 0, 20, &status);
    oskar_mem_set_value_real(two, 4.2, 0, 20, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ((int)OSKAR_TRUE, oskar_mem_different(one, two, 0, &status));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_free(one, &status);
    oskar_mem_free(two, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}


TEST(Mem, different_by_one)
{
    // Test two memory blocks that are different by one element.
    int status = 0;
    oskar_Mem *one = 0, *two = 0;
    one = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 20, &status);
    two = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 20, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_set_value_real(one, 1.0, 0, 20, &status);
    oskar_mem_set_value_real(two, 1.0, 0, 20, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_float(two, &status)[4] = 1.1f;
    ASSERT_EQ((int)OSKAR_TRUE, oskar_mem_different(one, two, 0, &status));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_free(one, &status);
    oskar_mem_free(two, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}


TEST(Mem, different_up_to_a_point)
{
    // Test two memory blocks that are different by one element, but only up to
    // the point where they are different.
    int status = 0;
    oskar_Mem *one = 0, *two = 0;
    one = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 20, &status);
    two = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 20, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_set_value_real(one, 1.0, 0, 20, &status);
    oskar_mem_set_value_real(two, 1.0, 0, 20, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_float(two, &status)[4] = 1.1f;
    ASSERT_EQ((int)OSKAR_FALSE, oskar_mem_different(one, two, 4, &status));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_free(one, &status);
    oskar_mem_free(two, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}
