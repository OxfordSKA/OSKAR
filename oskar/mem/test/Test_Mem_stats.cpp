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

#include "math.h"

        #include "mem/oskar_mem.h"
#include "utility/oskar_get_error_string.h"
#include <cmath>

TEST(Mem, stats)
{
    int status = 0;
    oskar_Mem* values = 0;
    values = oskar_mem_create(OSKAR_DOUBLE, OSKAR_CPU, 5, &status);

    // Fill an array with the values 1, 2, 3, 4, 5.
    double *v = oskar_mem_double(values, &status);
    v[0] = 1.0;
    v[1] = 2.0;
    v[2] = 3.0;
    v[3] = 4.0;
    v[4] = 5.0;

    // Compute minimum, maximum, mean and population standard deviation.
    double min = 0.0, max = 0.0, mean = 0.0, std_dev = 0.0;
    oskar_mem_stats(values, oskar_mem_length(values), &min, &max, &mean,
            &std_dev, &status);

    // Check values are correct.
    ASSERT_DOUBLE_EQ(3.0, mean);
    ASSERT_DOUBLE_EQ(1.0, min);
    ASSERT_DOUBLE_EQ(5.0, max);
    ASSERT_DOUBLE_EQ(sqrt(2.0), std_dev);

    // Free memory.
    oskar_mem_free(values, &status);
}

