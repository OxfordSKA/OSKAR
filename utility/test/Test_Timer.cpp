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

#include <oskar_timer.h>
#include <cstdlib>

#if 0
#include <unistd.h> // Needed for sleep() function.

TEST(Timer, test_consistency)
{
    oskar_Timer *t_cuda, *t_omp, *t_native;
    t_native = oskar_timer_create(OSKAR_TIMER_NATIVE);
    t_cuda = oskar_timer_create(OSKAR_TIMER_CUDA);
    t_omp = oskar_timer_create(OSKAR_TIMER_OMP);

    // Time a sleep(1).
    oskar_timer_resume(t_native);
    oskar_timer_resume(t_cuda);
    oskar_timer_resume(t_omp);
    sleep(1);
    oskar_timer_pause(t_native);
    oskar_timer_pause(t_cuda);
    oskar_timer_pause(t_omp);

    // Don't time this sleep.
    sleep(1);

    // Time another sleep(1).
    oskar_timer_resume(t_native);
    oskar_timer_resume(t_cuda);
    oskar_timer_resume(t_omp);
    sleep(1);
    oskar_timer_pause(t_native);
    oskar_timer_pause(t_cuda);
    oskar_timer_pause(t_omp);

    double elapsed_native = oskar_timer_elapsed(t_native);
    double elapsed_cuda = oskar_timer_elapsed(t_cuda);
    double elapsed_omp = oskar_timer_elapsed(t_omp);
    EXPECT_NEAR(2.0, elapsed_native, 1e-2);
#ifdef OSKAR_HAVE_CUDA
    EXPECT_NEAR(elapsed_native, elapsed_cuda, 5e-3);
    EXPECT_NEAR(2.0, elapsed_cuda, 1e-2);
#endif
#ifdef _OPENMP
    EXPECT_NEAR(elapsed_native, elapsed_omp, 5e-3);
    EXPECT_NEAR(2.0, elapsed_omp, 1e-2);
#endif

    oskar_timer_free(t_native);
    oskar_timer_free(t_cuda);
    oskar_timer_free(t_omp);
}
#endif

static void time_timer(int type, const char* label)
{
    oskar_Timer *tmr, *t;
    int runs = 1000;
    t = oskar_timer_create(OSKAR_TIMER_NATIVE);
    tmr = oskar_timer_create(type);
    oskar_timer_start(t);
    for (int i = 0; i < runs; ++i)
    {
        oskar_timer_resume(tmr);
        oskar_timer_pause(tmr);
    }
    printf("%s timer overhead: %.4e s.\n", label,
            oskar_timer_elapsed(t) / runs);
    oskar_timer_free(t);
    oskar_timer_free(tmr);
}

TEST(Timer, test_performance)
{
#ifdef OSKAR_HAVE_CUDA
    time_timer(OSKAR_TIMER_CUDA,   "  CUDA");
#endif
#ifdef _OPENMP
    time_timer(OSKAR_TIMER_OMP,    "OpenMP");
#endif
    time_timer(OSKAR_TIMER_NATIVE, "Native");
}
