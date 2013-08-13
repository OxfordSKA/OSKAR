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

#include "utility/test/Test_Timer.h"
#include "utility/oskar_timer_functions.h"
#include <unistd.h> // Note: this is a GNU compiler only header and needed
                    // for sleep() for some versions of g++ (tested with v 4.7)
                    // On windows this function is defined in windows.h
                    // with an argument in milliseconds rather than seconds.

void Test_Timer::test()
{
    oskar_Timer t_cuda, t_omp, t_native;
    oskar_timer_create(&t_native, OSKAR_TIMER_NATIVE);
    oskar_timer_create(&t_cuda, OSKAR_TIMER_CUDA);
    oskar_timer_create(&t_omp, OSKAR_TIMER_OMP);

    // Time a sleep(1).
    oskar_timer_resume(&t_native);
    oskar_timer_resume(&t_cuda);
    oskar_timer_resume(&t_omp);
    sleep(1);
    oskar_timer_pause(&t_native);
    oskar_timer_pause(&t_cuda);
    oskar_timer_pause(&t_omp);

    // Don't time this sleep.
    sleep(1);

    // Time another sleep(1).
    oskar_timer_resume(&t_native);
    oskar_timer_resume(&t_cuda);
    oskar_timer_resume(&t_omp);
    sleep(1);
    oskar_timer_pause(&t_native);
    oskar_timer_pause(&t_cuda);
    oskar_timer_pause(&t_omp);

    double elapsed_native = oskar_timer_elapsed(&t_native);
    double elapsed_cuda = oskar_timer_elapsed(&t_cuda);
    double elapsed_omp = oskar_timer_elapsed(&t_omp);
//    printf("Timings -- Native: %.5f, CUDA: %.5f, OpenMP: %.5f\n",
//            elapsed_native, elapsed_cuda, elapsed_omp);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(elapsed_native, elapsed_cuda, 5e-3);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(elapsed_native, elapsed_omp, 5e-3);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, elapsed_native, 1e-2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, elapsed_cuda, 1e-2);
    CPPUNIT_ASSERT_DOUBLES_EQUAL(2.0, elapsed_omp, 1e-2);

    oskar_timer_destroy(&t_native);
    oskar_timer_destroy(&t_cuda);
    oskar_timer_destroy(&t_omp);
}

void Test_Timer::test_performance()
{
    oskar_Timer t_native, t_cuda, t_omp, t;
    oskar_timer_create(&t, OSKAR_TIMER_NATIVE);
    oskar_timer_create(&t_native, OSKAR_TIMER_NATIVE);
    oskar_timer_create(&t_cuda, OSKAR_TIMER_CUDA);
    oskar_timer_create(&t_omp, OSKAR_TIMER_OMP);
    int runs = 1000;
    printf("\n");

    oskar_timer_start(&t);
    for (int i = 0; i < runs; ++i)
    {
        oskar_timer_resume(&t_cuda);
        oskar_timer_pause(&t_cuda);
    }
    printf("  CUDA timer overhead: %.4e s.\n", oskar_timer_elapsed(&t) / runs);

    oskar_timer_start(&t);
    for (int i = 0; i < runs; ++i)
    {
        oskar_timer_resume(&t_omp);
        oskar_timer_pause(&t_omp);
    }
    printf("OpenMP timer overhead: %.4e s.\n", oskar_timer_elapsed(&t) / runs);

    oskar_timer_start(&t);
    for (int i = 0; i < runs; ++i)
    {
        oskar_timer_resume(&t_native);
        oskar_timer_pause(&t_native);
    }
    printf("Native timer overhead: %.4e s.\n", oskar_timer_elapsed(&t) / runs);

    oskar_timer_destroy(&t);
    oskar_timer_destroy(&t_native);
    oskar_timer_destroy(&t_cuda);
    oskar_timer_destroy(&t_omp);
}
