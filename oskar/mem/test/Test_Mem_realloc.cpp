/*
 * Copyright (c) 2013-2017, The University of Oxford
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

#ifdef OSKAR_HAVE_CUDA
TEST(Mem, realloc_gpu)
{
    int status = 0;
    oskar_Mem *mem = oskar_mem_create(OSKAR_DOUBLE, OSKAR_GPU, 0, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_realloc(mem, 500, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(500, (int)oskar_mem_length(mem));
    ASSERT_EQ((int)OSKAR_DOUBLE, oskar_mem_type(mem));
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}
#endif


TEST(Mem, realloc_cpu)
{
    int status = 0;
    oskar_Mem *mem = oskar_mem_create(OSKAR_DOUBLE_COMPLEX, OSKAR_CPU,
            100, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    oskar_mem_realloc(mem, 1000, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    ASSERT_EQ(1000, (int)oskar_mem_length(mem));
    ASSERT_EQ((int)OSKAR_DOUBLE_COMPLEX, oskar_mem_type(mem));
    oskar_mem_free(mem, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
}
