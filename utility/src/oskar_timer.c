/*
 * Copyright (c) 2013-2015, The University of Oxford
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

#ifdef OSKAR_HAVE_CUDA
#include <cuda_runtime_api.h>
#endif

#include <private_timer.h>
#include <oskar_timer.h>
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#ifndef OSKAR_OS_WIN
#include <sys/time.h>
#include <unistd.h>
#else
#include <windows.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

static double oskar_get_wtime(oskar_Timer* timer)
{
    /* Declarations first (needs separate ifdef block). */
#ifdef OSKAR_OS_WIN
    LARGE_INTEGER cntr;
#else
#if _POSIX_MONOTONIC_CLOCK > 0
    struct timespec ts;
#else
    struct timeval tv;
#endif
#endif

    /* Return immediately if timer is not of native type. */
    if (timer->type != OSKAR_TIMER_NATIVE)
        return 0.0;

#ifdef OSKAR_OS_WIN
    /* Windows-specific version. */
    QueryPerformanceCounter(&cntr);
    return (double)(cntr.QuadPart) / timer->freq;
#else
#if _POSIX_MONOTONIC_CLOCK > 0
    /* Use monotonic clock if available. */
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec / 1e9;
#else
    /* Use gettimeofday() as fallback. */
    gettimeofday(&tv, 0);
    return tv.tv_sec + tv.tv_usec / 1e6;
#endif
#endif
}

oskar_Timer* oskar_timer_create(int type)
{
    oskar_Timer* timer;
#ifdef OSKAR_OS_WIN
    LARGE_INTEGER freq;
#endif

    /* Create the structure. */
    timer = malloc(sizeof(oskar_Timer));

#ifdef OSKAR_OS_WIN
    QueryPerformanceFrequency(&freq);
    timer->freq = (double)(freq.QuadPart);
#endif
    timer->type = type;
    timer->paused = 1;
    timer->elapsed = 0.0;
    timer->start = 0.0;
#ifdef OSKAR_HAVE_CUDA
    if (timer->type == OSKAR_TIMER_CUDA)
    {
        cudaEventCreate(&timer->start_cuda);
        cudaEventCreate(&timer->end_cuda);
    }
#endif
#ifdef _OPENMP
    omp_init_lock(&timer->mutex);
#endif
    return timer;
}

void oskar_timer_free(oskar_Timer* timer)
{
    if (!timer) return;
#ifdef OSKAR_HAVE_CUDA
    if (timer->type == OSKAR_TIMER_CUDA)
    {
        cudaEventDestroy(timer->start_cuda);
        cudaEventDestroy(timer->end_cuda);
    }
#endif
#ifdef _OPENMP
    omp_destroy_lock(&timer->mutex);
#endif
    free(timer);
}

double oskar_timer_elapsed(oskar_Timer* timer)
{
    double now = 0.0;

    /* If timer is paused, return immediately with current elapsed time. */
    if (timer->paused)
        return timer->elapsed;

#ifdef OSKAR_HAVE_CUDA
    /* CUDA timer. */
    if (timer->type == OSKAR_TIMER_CUDA)
    {
        float millisec = 0.0f;
#ifdef _OPENMP
        omp_set_lock(&timer->mutex); /* Lock the mutex */
#endif
        /* Get elapsed time since start. */
        cudaEventRecord(timer->end_cuda, 0);
        cudaEventSynchronize(timer->end_cuda);
        cudaEventElapsedTime(&millisec, timer->start_cuda, timer->end_cuda);

        /* Increment elapsed time. */
        timer->elapsed += millisec / 1000.0;

        /* Restart. */
        cudaEventRecord(timer->start_cuda, 0);
#ifdef _OPENMP
        omp_unset_lock(&timer->mutex); /* Unlock the mutex. */
#endif
        return timer->elapsed;
    }
#endif
#ifdef _OPENMP
    omp_set_lock(&timer->mutex); /* Lock the mutex */
    /* OpenMP timer. */
    if (timer->type == OSKAR_TIMER_OMP)
        now = omp_get_wtime();
#endif
    /* Native timer. */
    if (timer->type == OSKAR_TIMER_NATIVE)
        now = oskar_get_wtime(timer);

    /* Increment elapsed time and restart. */
    timer->elapsed += (now - timer->start);
    timer->start = now;
#ifdef _OPENMP
    omp_unset_lock(&timer->mutex); /* Unlock the mutex. */
#endif
    return timer->elapsed;
}

void oskar_timer_pause(oskar_Timer* timer)
{
    if (timer->paused)
        return;
    (void)oskar_timer_elapsed(timer);
    timer->paused = 1;
}

void oskar_timer_resume(oskar_Timer* timer)
{
    if (!timer->paused)
        return;
    oskar_timer_restart(timer);
}

void oskar_timer_restart(oskar_Timer* timer)
{
    timer->paused = 0;
#ifdef OSKAR_HAVE_CUDA
    /* CUDA timer. */
    if (timer->type == OSKAR_TIMER_CUDA)
    {
        cudaEventRecord(timer->start_cuda, 0);
        return;
    }
#endif
#ifdef _OPENMP
    /* OpenMP timer. */
    if (timer->type == OSKAR_TIMER_OMP)
    {
        timer->start = omp_get_wtime();
        return;
    }
#endif
    /* Native timer. */
    timer->start = oskar_get_wtime(timer);
}

void oskar_timer_start(oskar_Timer* timer)
{
    timer->elapsed = 0.0;
    oskar_timer_restart(timer);
}

#ifdef __cplusplus
}
#endif
