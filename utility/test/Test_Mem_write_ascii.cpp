/*
 * Copyright (c) 2013, The University of Oxford
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

#include <oskar_get_error_string.h>
#include <oskar_mem.h>
#include <oskar_timer.h>
#include <cstdio>

TEST(Mem, write_ascii)
{
    int status = 0;
    oskar_Mem mem1, mem2, mem3, mem4, mem5, mem6, mem7, mem8;
    size_t length = 100;
    oskar_mem_init(&mem1, OSKAR_SINGLE, OSKAR_LOCATION_CPU, length, 1, &status);
    oskar_mem_init(&mem2, OSKAR_DOUBLE, OSKAR_LOCATION_CPU, length, 1, &status);
    oskar_mem_init(&mem3, OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_CPU, length, 1, &status);
    oskar_mem_init(&mem4, OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_CPU, length, 1, &status);
    oskar_mem_init(&mem5, OSKAR_SINGLE, OSKAR_LOCATION_GPU, length, 1, &status);
    oskar_mem_init(&mem6, OSKAR_DOUBLE, OSKAR_LOCATION_GPU, length, 1, &status);
    oskar_mem_init(&mem7, OSKAR_SINGLE_COMPLEX, OSKAR_LOCATION_GPU, length, 1, &status);
    oskar_mem_init(&mem8, OSKAR_DOUBLE_COMPLEX, OSKAR_LOCATION_GPU, length, 1, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    oskar_mem_set_value_real(&mem1, 1.0, 0, 0, &status);
    oskar_mem_set_value_real(&mem2, 2.0, 0, 0, &status);
    oskar_mem_set_value_real(&mem3, 3.0, 0, 0, &status);
    oskar_mem_set_value_real(&mem4, 4.0, 0, 0, &status);
    oskar_mem_set_value_real(&mem5, 5.0, 0, 0, &status);
    oskar_mem_set_value_real(&mem6, 6.0, 0, 0, &status);
    oskar_mem_set_value_real(&mem7, 7.0, 0, 0, &status);
    oskar_mem_set_value_real(&mem8, 8.0, 0, 0, &status);

    const char* fname = "temp_test_wrtie_ascii.txt";
    FILE* f = fopen(fname, "w");
//    oskar_Timer* timer = oskar_timer_create(OSKAR_TIMER_NATIVE);
//    oskar_timer_start(timer);
    oskar_mem_write_ascii(f, 8, length, &status,
            &mem1, &mem2, &mem3, &mem4, &mem5, &mem6, &mem7, &mem8);
//    printf("Elapsed time: %.5f sec\n", oskar_timer_elapsed(timer));
//    oskar_timer_free(timer);
    fclose(f);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);

    oskar_mem_free(&mem1, &status);
    oskar_mem_free(&mem2, &status);
    oskar_mem_free(&mem3, &status);
    oskar_mem_free(&mem4, &status);
    oskar_mem_free(&mem5, &status);
    oskar_mem_free(&mem6, &status);
    oskar_mem_free(&mem7, &status);
    oskar_mem_free(&mem8, &status);
    ASSERT_EQ(0, status) << oskar_get_error_string(status);
    remove(fname);
}

