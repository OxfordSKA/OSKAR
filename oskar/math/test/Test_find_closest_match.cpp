/*
 * Copyright (c) 2012-2014, The University of Oxford
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

#include <oskar_find_closest_match.h>
#include <oskar_mem.h>
#include <oskar_get_error_string.h>

TEST(find_closest_match, test)
{
    int i = 0, size = 10, status = 0;
    int type = OSKAR_DOUBLE, location = OSKAR_CPU;
    double start = 0.0, inc = 0.3, value = 0.0, *values_;
    oskar_Mem* values;

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
