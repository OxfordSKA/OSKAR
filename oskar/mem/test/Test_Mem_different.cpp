/*
 * Copyright (c) 2013-2014, The University of Oxford
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


TEST(Mem, different_none)
{
    // Test two memory blocks that are the same.
    int status = 0;
    oskar_Mem *one, *two;
    one = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 20, &status);
    two = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 20, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_set_value_real(one, 4.4, 0, 0, &status);
    oskar_mem_set_value_real(two, 4.4, 0, 0, &status);
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
    oskar_Mem *one, *two;
    one = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 20, &status);
    two = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 20, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_set_value_real(one, 4.4, 0, 0, &status);
    oskar_mem_set_value_real(two, 4.2, 0, 0, &status);
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
    oskar_Mem *one, *two;
    one = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 20, &status);
    two = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 20, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_set_value_real(one, 1.0, 0, 0, &status);
    oskar_mem_set_value_real(two, 1.0, 0, 0, &status);
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
    oskar_Mem *one, *two;
    one = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 20, &status);
    two = oskar_mem_create(OSKAR_SINGLE, OSKAR_CPU, 20, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_set_value_real(one, 1.0, 0, 0, &status);
    oskar_mem_set_value_real(two, 1.0, 0, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_float(two, &status)[4] = 1.1f;
    ASSERT_EQ((int)OSKAR_FALSE, oskar_mem_different(one, two, 4, &status));
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_free(one, &status);
    oskar_mem_free(two, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}

