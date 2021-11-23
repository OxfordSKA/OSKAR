/*
 * Copyright (c) 2012-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "math/oskar_find_closest_match.h"
#include "mem/oskar_mem.h"
#include "utility/oskar_get_error_string.h"

TEST(find_closest_match, test)
{
    int i = 0, size = 10, status = 0;
    int type = OSKAR_DOUBLE, location = OSKAR_CPU;
    double start = 0.0, inc = 0.3, value = 0.0, *values_ = 0;
    oskar_Mem* values = 0;

    // Create array and fill with values.
    values = oskar_mem_create(type, location, size, &status);
    values_ = oskar_mem_double(values, &status);
    for (i = 0; i < size; ++i)
    {
        values_[i] = start + inc * i;
    }

    //  0    1    2    3    4    5    6    7    8    9
    // 0.0  0.3  0.6  0.9  1.2  1.5  1.8  2.1  2.4  2.7

    value = 0.7;
    i = oskar_find_closest_match(value, values, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(2, i);

    value = 0.749999;
    i = oskar_find_closest_match(value, values, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(2, i);

    value = 0.75;
    i = oskar_find_closest_match(value, values, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(3, i);

    value = 0.750001;
    i = oskar_find_closest_match(value, values, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(3, i);

    value = 100;
    i = oskar_find_closest_match(value, values, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(9, i);

    value = -100;
    i = oskar_find_closest_match(value, values, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(0, i);

    value = 0.3;
    i = oskar_find_closest_match(value, values, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    EXPECT_EQ(1, i);

    // Free memory.
    oskar_mem_free(values, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}
