/*
 * Copyright (c) 2013-2021, The OSKAR Developers.
 * See the LICENSE file at the top-level directory of this distribution.
 */

#include <gtest/gtest.h>

#include "utility/oskar_timer.h"
#include <cstdlib>

#if 0
#include <unistd.h> // Needed for sleep() function.

TEST(Timer, test_consistency)
{
    oskar_Timer *t_cuda, *t_omp, *t_native;
    t_native = oskar_timer_create(OSKAR_TIMER_NATIVE);
    t_cuda = oskar_timer_create(OSKAR_TIMER_CUDA);

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
    oskar_Timer *tmr = 0, *t = 0;
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
#endif
    time_timer(OSKAR_TIMER_NATIVE, "Native");
}
